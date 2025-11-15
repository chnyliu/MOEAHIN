import torch.nn as nn
from torch_geometric.nn.conv import GATConv, GCNConv, ChebConv, SAGEConv, APPNP, ARMAConv, EdgeConv
from torch.nn import Sequential, Linear, ReLU
from utils import cstr_nc, gnn_search_space
import torch.nn.functional as F
import torch
import torch_geometric as pyg


def gnn_map(gnn_name, in_dim, out_dim, dropout, bias=True):
    """
    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param dropout:
    :param concat: for gat, concat multi-head output or not
    :param bias:
    :return: GNN model
    """

    if gnn_name == "GAT_8":
        return GATConv(in_dim, out_dim, 8, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "GAT_4":
        return GATConv(in_dim, out_dim, 4, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "GAT_1":
        return GATConv(in_dim, out_dim, 1, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "GCN":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "Cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "SAGE_MEAN":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='mean')
    elif gnn_name == "SAGE_MAX":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='max')
    elif gnn_name == "SAGE_SUM":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='add')
    elif gnn_name == "ARMA":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "APPNP":
        return APPNP(K=10, alpha=0.1)
    elif gnn_name == 'EDGE':
        nn1 = Linear(in_dim*2, out_dim)
        return EdgeConv(nn1)
    else:
        raise Exception("wrong gnn name")


def act_map(act):
    if act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return F.relu
    elif act == "leaky":
        return F.leaky_relu
    elif act == "gelu":
        return F.gelu
    else:
        raise Exception("wrong activate function")


class HGNN(nn.Module):
    def __init__(self, meta_graph, gnn_arch, layer_num, adj_len, num_node_types, in_feat, hidden_dim, out_classes,
                 dropout, layer_agg, act, bias=True, use_norm=True):
        super(HGNN, self).__init__()
        self.num_node_types = num_node_types
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cstr = cstr_nc

        self.in_layer = nn.ModuleList()
        for i in range(self.num_node_types):
            self.in_layer.append(nn.Linear(in_feat, hidden_dim))

        self.layers = nn.ModuleList()
        self.paths = []
        index = 0
        for i in range(layer_num):  # 第i=0层
            layer = nn.ModuleList()
            path = []
            for j in range(index, index + (i + 1) * adj_len):
                if meta_graph[j] == 1:
                    layer.append(gnn_map(gnn_search_space[gnn_arch[j] - 1], hidden_dim, hidden_dim, dropout, bias))
                    path.append([(j - index) // adj_len, j % adj_len])
            index += (i + 1) * adj_len
            self.layers.append(layer)
            self.paths.append(path)

        self.out_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer2 = nn.Linear(hidden_dim, 1)
        self.out_layer3 = nn.Linear(hidden_dim, out_classes)
        self.layer_agg = layer_agg
        self.activation = act_map(act)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False) if use_norm is True else lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            for aggr in layer:
                aggr.reset_parameters()

    def forward(self, x, adjs_pt, node_types):
        try:
            hid = torch.zeros((node_types.size(0), self.hidden_dim)).cuda()
            for i in range(self.num_node_types):
                idx = (node_types == i)
                hid[idx] = self.in_layer[i](x[idx].cuda())
            # x = self.activation(hid)
            x = hid

            states = [x]
            for i in range(self.layer_num):
                if len(self.paths[i]) == 0:  # 这一层为空
                    h = torch.FloatTensor(size=states[0].shape).cuda()
                else:
                    h = []
                    paths_i_len = len(self.paths[i])
                    for j in range(paths_i_len):
                        agg = self.layers[i][j]
                        path = self.paths[i][j]
                        (edge_index, _) = pyg.utils.to_edge_index(adjs_pt[path[1]].coalesce().cuda())
                        _h = F.dropout(states[path[0]], p=self.dropout, training=self.training)
                        h.append(agg(_h, edge_index.cuda()))
                    if self.layer_agg == 'max':
                        h = torch.stack(h, dim=0)
                        h = torch.max(h, dim=0)[0]
                    elif self.layer_agg == 'sum':
                        h = torch.stack(h, dim=0)
                        h = torch.sum(h, dim=0)
                    elif self.layer_agg == 'mean':
                        h = torch.stack(h, dim=0)
                        h = torch.mean(h, dim=0)
                    else:
                        raise
                # states.append(self.activation(h))
                states.append(h)
            tmp = states[-1]  #18405, 64
            tmp = self.norm(tmp)
            hidi = F.gelu(tmp)  #18405, 64
            temps = [hidi]
            attni = self.out_layer2(torch.tanh(self.out_layer1(temps[-1])))  # 18405, 1
            attns = [attni]
            hids = torch.stack(temps, dim=0).transpose(0, 1)  # 18405, 1, 64
            attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)  # 18405, 1
            out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)  # 18405， 64
            return self.out_layer3(out)
        except RuntimeError as e:
            raise
