import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hgnn import HGNN, HyperSAGE, UniGNN
from dgi import DGI, DGI_SAGE


class Predictor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn = 'mse',
            activation = 'prelu',
         ):
        super(Predictor, self).__init__()
        
        dropout = 0
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim))        
        self.mod = HGNN(
            in_dim=hidden_dim, 
            num_hidden=hidden_dim, 
            out_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation)
        self.linear = nn.Linear(hidden_dim, out_dim)
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")
    
    
    def forward(self, H, he_rep, x):      
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, H)
        x_prime = self.linear(F.leaky_relu(enc))
        loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    
    def predict(self, H, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, H)
        x_prime = self.linear(F.leaky_relu(enc))
        return x_prime

class Predictor_SAGE(nn.Module):
    def __init__(
            self,
            hyper_graph,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn = 'mse',
            device = 'cpu'
         ):
        super(Predictor_SAGE, self).__init__()
        
        dropout = 0
        self.device = device
        self.hyper_graph = hyper_graph
        self.node_by_node = hyper_graph @ hyper_graph.T
        self.gnn_layers = num_layers
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(hidden_dim))        
        self.gnn = HyperSAGE(hyper_graph = hyper_graph,
                             in_dim = hidden_dim, 
                             hidden_dim = hidden_dim, 
                             out_dim = hidden_dim, 
                             num_layers = num_layers, 
                             dropout = dropout,
                             device = device)
        self.dgi = DGI_SAGE(hyper_graph=hyper_graph,
                            num_layers=2,
                            dropout=dropout,
                            device=device,
                            in_dim=in_dim,
                            hidden_dim=hidden_dim)
        self.predicter = nn.Linear(hidden_dim, out_dim)
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")
    
    def get_neighbors(self, tgt_idx, hyper_graph=None):
        if hyper_graph is not None:
            node_by_node = hyper_graph @ hyper_graph.T
        else:
            node_by_node = self.node_by_node

        neighbor_list = [(tgt_idx, None), ]
        for _ in range(self.gnn_layers):
            tgt_idx = np.unique(node_by_node[tgt_idx].tocoo().col)                                # 找到邻居节点
            mapped_indices = torch.arange(tgt_idx.shape[0], device=self.device, dtype=torch.int32)
            mapping = torch.zeros(node_by_node.shape[0], dtype=torch.int32, device=self.device)
            mapping[tgt_idx] = mapped_indices
            neighbor_list.append((tgt_idx, mapping))
        return neighbor_list
    
    def forward(self, tgt_id, node_feat, x):     
        neighbor_list = self.get_neighbors(tgt_id) 
        embed = self.mlp(node_feat[neighbor_list[-1][0]].to(self.device))                         # 对所有参与训练的节点计算
        embed = self.gnn(embed, neighbor_list)
        x_prime = self.predicter(F.leaky_relu(embed))
        loss = self.criterion(x_prime, x[tgt_id].to(self.device))
        return loss, x_prime, embed

    def predict(self, tgt_id, node_feat, hyper_graph=None):
        neighbor_list = self.get_neighbors(tgt_id, hyper_graph) 
        embed = self.mlp(node_feat[neighbor_list[-1][0]].to(self.device))
        embed = self.gnn.predict(hyper_graph, embed, neighbor_list)
        x_prime = self.predicter(F.leaky_relu(embed))
        return x_prime
    
    
class Predictor_dgi(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
         ):
        super(Predictor_dgi, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)
            )
        self.dgi = DGI(hidden_dim, hidden_dim, out_dim, 'prelu')    
        self.b_xent = nn.CosineEmbeddingLoss()
    
    
    def forward(self, H, x):      
        h = self.mlp(x)

        nb_nodes = x.shape[0]
        idx = torch.randperm(nb_nodes)
        shuf_fts = h[idx, :]
    
        lbl_1 = torch.ones(nb_nodes).to(x.device)
        lbl_2 = -torch.ones(nb_nodes).to(x.device)
        
        h1, h2, c = self.dgi(h, shuf_fts, H)
        
        loss = self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)

        return loss


class Predictor_uniSAGE(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn = 'mse',
            activation = 'prelu',
         ):
        super(Predictor, self).__init__()
        
        dropout = 0
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim))        
        self.mod = UniGNN(
            in_dim=hidden_dim, 
            num_hidden=hidden_dim, 
            out_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation)
        self.linear = nn.Linear(hidden_dim, out_dim)
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")
    
    
    def forward(self, H, he_rep, x):      
        he_rep = self.mlp(he_rep)
        V, E = H.coalesce().indices()
        enc = self.mod(he_rep, V, E)
        x_prime = self.linear(F.leaky_relu(enc))
        loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    
    def predict(self, H, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, H)
        x_prime = self.linear(F.leaky_relu(enc))
        return x_prime
