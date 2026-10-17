import math

import torch
from torch import nn
from torch.nn import functional as F


def _sum_to_dst(graph, message):
    out = message.new_zeros((graph.num_nodes(),) + message.shape[1:])
    return out.index_add_(0, graph.dst, message)


def _degree(index, n):
    return torch.bincount(index, minlength=n).clamp(min=1).float()


class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, graph, feat):
        n = graph.num_nodes()
        feat = feat * _degree(graph.src, n).pow(-0.5).unsqueeze(1)
        if self.weight.shape[0] > self.weight.shape[1]:
            feat = feat @ self.weight
            rst = _sum_to_dst(graph, feat[graph.src])
        else:
            rst = _sum_to_dst(graph, feat[graph.src]) @ self.weight
        rst = rst * _degree(graph.dst, n).pow(-0.5).unsqueeze(1) + self.bias
        return self.activation(rst) if self.activation else rst


class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation=None, negative_slope=0.2):
        super().__init__()
        self.num_heads, self.out_feats, self.negative_slope = num_heads, out_feats, negative_slope
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.bias = nn.Parameter(torch.zeros(num_heads * out_feats))
        gain = nn.init.calculate_gain('relu')
        for p in (self.fc.weight, self.attn_l, self.attn_r):
            nn.init.xavier_normal_(p, gain=gain)
        self.activation = activation

    def forward(self, graph, feat):
        n = graph.num_nodes()
        feat = self.fc(feat).view(n, self.num_heads, self.out_feats)
        el = (feat * self.attn_l).sum(-1)
        er = (feat * self.attn_r).sum(-1)
        e = F.leaky_relu(el[graph.src] + er[graph.dst], self.negative_slope)
        index = graph.dst.unsqueeze(1).expand_as(e)
        emax = e.new_full((n, self.num_heads), -torch.inf).scatter_reduce(0, index, e, 'amax')
        e = torch.exp(e - emax[graph.dst])
        a = e / _sum_to_dst(graph, e)[graph.dst]
        rst = _sum_to_dst(graph, feat[graph.src] * a.unsqueeze(-1))
        rst = rst + self.bias.view(1, self.num_heads, self.out_feats)
        return self.activation(rst) if self.activation else rst


class SAGEConv(nn.Module):
    """GraphSAGE with the max-pooling aggregator."""

    def __init__(self, in_feats, out_feats, activation=None):
        super().__init__()
        self.fc_pool = nn.Linear(in_feats, in_feats)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.activation = activation

    def forward(self, graph, feat):
        msg = F.relu(self.fc_pool(feat))[graph.src]
        index = graph.dst.unsqueeze(1).expand_as(msg)
        neigh = msg.new_zeros((graph.num_nodes(), msg.shape[1])).scatter_reduce(0, index, msg, 'amax', include_self=False)
        rst = self.fc_self(feat) + self.fc_neigh(neigh) + self.bias
        return self.activation(rst) if self.activation else rst


class TAGConv(nn.Module):
    def __init__(self, in_feats, out_feats, k=2, activation=None):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(in_feats * (k + 1), out_feats)
        self.activation = activation

    def forward(self, graph, feat):
        norm = _degree(graph.dst, graph.num_nodes()).pow(-0.5).unsqueeze(1)
        stack = [feat]
        for _ in range(self.k):
            stack.append(_sum_to_dst(graph, (stack[-1] * norm)[graph.src]) * norm)
        rst = self.lin(torch.cat(stack, dim=-1))
        return self.activation(rst) if self.activation else rst


class SGConv(nn.Module):
    def __init__(self, in_feats, out_feats, k=1):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(in_feats, out_feats)

    def forward(self, graph, feat):
        n = graph.num_nodes()
        norm_src = _degree(graph.src, n).pow(-0.5).unsqueeze(1)
        norm_dst = _degree(graph.dst, n).pow(-0.5).unsqueeze(1)
        for _ in range(self.k):
            feat = _sum_to_dst(graph, (feat * norm_src)[graph.src]) * norm_dst
        return self.fc(feat)


class SumPooling(nn.Module):
    def forward(self, graph, feat):
        batch = torch.repeat_interleave(torch.arange(len(graph.node_counts), device=feat.device), graph.node_counts)
        return feat.new_zeros((len(graph.node_counts), feat.shape[1])).index_add_(0, batch, feat)
