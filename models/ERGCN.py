# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from torch_geometric.nn import GATConv
import scipy.sparse as sp
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # torch.Size([32, 55, 512]) torch.Size([512, 512]) torch.Size([32, 55, 512])
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # torch.Size([32, 69, 1]) torch.Size([32, 69, 69])
        output = torch.matmul(adj, hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ERGCN(nn.Module):
    def __init__(self, bert, opt):
        super(ERGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)

        # bert_feature 768维变为512维
        self.text_lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # vit的维度768维变为512维
        self.vit_fc = nn.Linear(768, 2*opt.hidden_dim)
        # 将512维转化为2维  true or false
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)


    def forward(self, inputs):
        bert_indices, graph, box_vit = inputs   # torch.Size([32, 45]) torch.Size([32, 69, 69]) torch.Size([32, 10, 768])
        adj = sp.coo_matrix(graph)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        graph = torch.LongTensor(indices)  # PyG框架需要的coo形式

        bert_text_len = torch.sum(bert_indices != 0, dim=-1)    # torch.Size([32])
        encoder_layer, pooled_output = self.bert(bert_indices,  output_all_encoded_layers=False)
        text_out, (_, _) = self.text_lstm(encoder_layer, bert_text_len)
        # text out  torch.Size([32, 45, 512])

        #caption_text_len = torch.sum(caption_indices != 0, dim=-1)
        #caption_encoder_layer, caption_pooled_output = self.bert(caption_indices, output_all_encoded_layers=False)
        #caption_text_out, (_, _) = self.text_lstm(caption_encoder_layer, caption_text_len)
        box_vit = box_vit.to(torch.float32)   # 这个不能缺少
        box_vit = self.vit_fc(box_vit)   # torch.Size([32, 10, 512])
        features = torch.cat([text_out, box_vit], dim = 1)
        # torch.Size([32, 55, 512])
        x = F.relu(self.gc1(features, graph))
        self.relu = F.relu(self.gc2(x, graph))
        x = self.relu
        # 下面好像就是attention部分
        alpha_mat = torch.matmul(features, x.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, x).squeeze(1)
        output = self.fc(x)
        return output
