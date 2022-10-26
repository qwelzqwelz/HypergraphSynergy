import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
import sys

sys.path.append('..')
from utils import reset

from typing import Optional
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


drug_num = 87
cline_num = 55

class PrintableHypergraphConv(HypergraphConv):
    
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        ###############################################################################
        D_zero_index = (D == 0)
        print("D_zero_index", D_zero_index.shape)
        print("D_zero_index", D_zero_index)
        print("D_zero_index", [i for i, x in enumerate(D_zero_index) if x == True])
        print("hp-x", x.shape)
        print("hp-x", x[D_zero_index,:])
        ###############################################################################

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out



class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = PrintableHypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = PrintableHypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = PrintableHypergraphConv(256, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.conv1(x, edge)
        print("conv1-result[130]:", x[130,:])
        print("conv1-result[131]:", x[131,:])
        x = self.batch1(self.act(x))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.4)
        self.act = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h1 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=1))


class HypergraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder):
        super(HypergraphSynergy, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.drug_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.cline_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        print("graph_embed[130]", graph_embed[130,:])
        print("graph_embed[131]", graph_embed[131,:])
        drug_emb, cline_emb = graph_embed[:drug_num], graph_embed[drug_num:]
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res, rec_drug, rec_cline
