import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from utils import create_activation


class HGNN(nn.Module):
    def __init__(self,                  
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation):
                 
        super(HGNN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = create_activation(activation)
        self.mlp = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.W1 = nn.Linear(in_dim, out_dim)
        elif num_layers == 2:
            self.W1 = nn.Linear(in_dim, num_hidden)
            self.W2 = nn.Linear(num_hidden, out_dim)
        elif self.num_layers > 2:
            for i in range(self.num_layers-2):
                self.mlp.append(nn.Linear(num_hidden, num_hidden))
            
        self.dropout = nn.Dropout(dropout)


    def forward(self, X, H):
        if self.num_layers == 1:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
        elif self.num_layers == 2:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))
        else:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            for i in range(self.num_layers-2):
                X = torch.sparse.mm(H, self.mlp[i](self.dropout(X)))
                X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))
            
        return X
    

class HyperSAGE(nn.Module):
    def __init__(self,                
                 in_dim, 
                 hidden_dim, 
                 out_dim,
                 num_layers,
                 dropout,
                 device):
        super(HyperSAGE, self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if self.num_layers > 1:
            self.weight_list = [nn.Parameter(torch.FloatTensor(2*in_dim, hidden_dim)).to(self.device), ]
            for _ in range(self.num_layers - 2):
                self.weight_list.append(nn.Parameter(torch.FloatTensor(2*hidden_dim, hidden_dim)).to(self.device))
            self.weight_list.append(nn.Parameter(torch.FloatTensor(2*hidden_dim, out_dim)).to(self.device))
        else:
            self.weight_list = [nn.Parameter(torch.FloatTensor(2*in_dim, out_dim)).to(self.device), ]
        for weight in self.weight_list:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, node_feat, neighbor_list, graph_attr):
        '''训练过程中会提前计算好一些图属性以加速训练'''
        for layer in range(self.num_layers):
            neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers-layer-1][0], neighbor_list[self.num_layers-layer][0], node_feat, graph_attr)
            map_dict = neighbor_list[self.num_layers-layer][1]
            tgt_index = map_dict[neighbor_list[self.num_layers-1-layer][0]]                     # 全局索引映射回上一层索引
            feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
            node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight_list[layer]))
        return node_feat
    
    def predict(self, node_feat, neighbor_list, graph_attr):
        '''需要重新计算超图相关属性'''
        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers-1][0], neighbor_list[self.num_layers][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers][1]
        tgt_index = map_dict[neighbor_list[self.num_layers-1][0]]                     
        feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
        node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight1))

        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers-2][0], neighbor_list[self.num_layers-1][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers-1][1]
        tgt_index = map_dict[neighbor_list[self.num_layers-2][0]]
        feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
        node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight2))
        return node_feat
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx, cuda=False):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        if cuda:
            return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        else:
            return  torch.sparse.FloatTensor(indices, values, shape)

    def sparse_diags(self, data):
        indices = torch.arange(data.shape[0])
        indices = torch.vstack([indices, indices]).to(self.device)
        return torch.sparse_coo_tensor(indices, data, (data.shape[0], data.shape[0]))
        

    def initialize(self, hyper_graph):
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()                                                    # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()                                                    # 每个节点被几个超边包含
        
        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)    # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()                                               # 计算每个节点的邻居节点数量
        return num_nodes, num_edges, num_neighbors

    def Aggregate_neighbors(self, tgt_idx, src_idx, node_emb, graph_attr=None):
        '''
        前向传播每次需要聚合邻居节点
        tgt_idx,            【list】,             本层需要更新的目标节点
        src_idx,            【list】,             本层目标节点的邻居节点
        node_emb,           【torch.Tensor, 2D】, 前一层的节点表征
        model,              【str】             , 
        '''

        hyper_graph, num_nodes, num_edges, num_neighbors = graph_attr['graph'], graph_attr['num_nodes'], graph_attr['num_edges'], graph_attr['num_neighbors']

        '''将源节点特征聚合到超边上'''
        tgt_edge = np.unique(hyper_graph[tgt_idx].tocoo().col)                                         # 相关超边，只有稀疏array支持索引
        edge_cardinality = num_nodes[tgt_edge]
        edge_cardinality_inv = 1.0 / edge_cardinality    
        edge_cardinality = self.sparse_diags(edge_cardinality)
        edge_cardinality_inv = self.sparse_diags(edge_cardinality_inv)
        
        edge_agg_mtx = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[src_idx][:, tgt_edge].T, cuda=True)      # 目标节点参与的超边*邻接节点的聚合矩阵
        edge_emb = torch.spmm(torch.spmm(edge_cardinality_inv, edge_agg_mtx), node_emb)                # 将节点特征聚合到超边上

        '''将超边特征聚合到目标节点上'''
        num_neighbor_inv = 1.0 / num_neighbors[tgt_idx]
        num_neighbor_inv = self.sparse_diags(num_neighbor_inv)  # 算目标节点所有通过超边邻接的节点数量
        num_edge_inv = 1.0 / num_edges[tgt_idx]
        num_edge_inv = self.sparse_diags(num_edge_inv)

        tgt_by_edge = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[tgt_idx][:, tgt_edge], cuda=True)
        neighbor_agg_emb = torch.spmm(num_edge_inv, num_neighbor_inv @ torch.spmm(torch.spmm(tgt_by_edge, edge_cardinality), edge_emb))
        return neighbor_agg_emb


class UniGNN(nn.Module):
    def __init__(self,                  
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation):
                 
        super(HGNN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = create_activation(activation)
        self.mlp = nn.ModuleList()
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim)
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean') # [E, C]

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = X + Xv 

        return X