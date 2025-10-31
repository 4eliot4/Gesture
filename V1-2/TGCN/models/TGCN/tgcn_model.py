#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np


class GraphConvolution_att(nn.Module):
    """
    Simple GCN layer similar to https://arxiv.org/abs/1609.02907, with an attention matrix.
    Now, the number of nodes is parameterized via `num_nodes`.
    """
    def __init__(self, in_features, out_features, num_nodes=55, bias=True):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        
        # Weight matrix: transforms each node's feature from in_features to out_features.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Attention (or support) matrix: should have shape (num_nodes, num_nodes).
        self.att = Parameter(torch.FloatTensor(num_nodes, num_nodes))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # Input shape: (batch, num_nodes, in_features)
        # Multiply each node's features by the weight matrix: result shape: (batch, num_nodes, out_features)
        support = torch.matmul(input, self.weight)
        # Multiply the support matrix by the attention (graph) matrix:
        # self.att: (num_nodes, num_nodes) and support: (batch, num_nodes, out_features)
        # Broadcasting allows: output: (batch, num_nodes, out_features)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}, num_nodes={self.num_nodes})"


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, num_nodes=55, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi
        self.num_nodes = num_nodes

        # Use num_nodes in the BatchNorm dimensions.
        self.gc1 = GraphConvolution_att(in_features, in_features, num_nodes=num_nodes, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_nodes * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features, num_nodes=num_nodes, bias=bias)
        self.bn2 = nn.BatchNorm1d(num_nodes * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}, num_nodes={self.num_nodes})"


class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True, num_nodes=55):
        """
        Args:
            input_feature: number of features per node in the input.
            hidden_feature: hidden dimension used in GCN layers.
            num_class: number of output classes.
            p_dropout: dropout probability.
            num_stage: number of GC_Block stages.
            is_resi: whether to use residual connections in GC_Block.
            num_nodes: number of nodes in the graph (e.g. 33 for 33 landmarks).
        """
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage
        self.num_nodes = num_nodes

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature, num_nodes=num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, num_nodes=num_nodes, is_resi=is_resi))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        # Final fully-connected layer applied after pooling over nodes.
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        # x expected shape: (batch, num_nodes, input_feature)
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        # Pool over nodes (e.g., take the mean)
        out = torch.mean(y, dim=1)
        out = self.fc_out(out)
        return out
    


class GCN_muti_att_Model(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, num_classes, dropout_rate, 
                 num_gcn_stages=1, use_residual_gcn=True, num_graph_nodes=75, pooling_type='flatten'):
        super(GCN_muti_att_Model, self).__init__()
        self.num_graph_nodes = num_graph_nodes
        self.pooling_type = pooling_type.lower()
        self.gc1 = GraphConvolution_att(input_feature_dim, hidden_feature_dim, num_nodes=num_graph_nodes)
        self.bn1 = nn.BatchNorm1d(num_graph_nodes * hidden_feature_dim)
        self.gc_blocks = nn.ModuleList()
        for _ in range(num_gcn_stages):
            self.gc_blocks.append(GC_Block(hidden_feature_dim, p_dropout=dropout_rate, 
                                           num_nodes=num_graph_nodes, is_resi=use_residual_gcn))
        self.dropout_after_gcn = nn.Dropout(dropout_rate)
        self.activation_after_gcn = nn.Tanh()
        if self.pooling_type == 'mean': self.fc_out = nn.Linear(hidden_feature_dim, num_classes)
        elif self.pooling_type == 'flatten': self.fc_out = nn.Linear(hidden_feature_dim * num_graph_nodes, num_classes)
        else: raise ValueError(f"Unsupported pooling_type: {pooling_type}")
    def forward(self, x):
        b, n, _ = x.shape
        y = self.gc1(x); y = self.bn1(y.view(b, -1)).view(b, n, -1)
        y = self.activation_after_gcn(y); y = self.dropout_after_gcn(y)
        for block in self.gc_blocks: y = block(y)
        out_pooled = torch.mean(y, dim=1) if self.pooling_type == 'mean' else y.contiguous().view(b, -1)
        return self.fc_out(out_pooled)
# --- End Model Definitions ---


class GCN_muti_att_Mod(nn.Module):
    """
    Modified version of TGCN model wrapper.
    In inference if only raw features (8 channels per node) are provided, duplicate them to mimic 16 channels (for data augmentation).
    """
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage, num_nodes):
        super(GCN_muti_att_Mod, self).__init__()
        self.gcn = GCN_muti_att(
            input_feature=input_feature,
            hidden_feature=hidden_feature,
            num_class=num_class,
            p_dropout=p_dropout,
            num_stage=num_stage,
            num_nodes=num_nodes
        )
    
    def forward(self, x):
        if x.size(-1) == 8:
            x = torch.cat([x, x], dim=-1)
        return self.gcn(x)
