import torch
import torch.nn as nn
from hgnn import HGNN, HyperSAGE
from utils import create_activation


class DGI(nn.Module):
    def __init__(self, n_in, n_hid, n_h, activation):
        super(DGI, self).__init__()

        self.hgnn = HGNN(in_dim=n_in,
            num_hidden=n_hid,
            out_dim=n_h, 
            num_layers=1, 
            dropout=0.1, 
            activation='prelu')

    def forward(self, seq1, seq2, adj):
        h1 = self.hgnn(seq1, adj)          # 每个细胞正确的表征
 
        c = torch.mean(h1, dim=0)          # 正确的全局表征

        h2 = self.hgnn(seq2, adj)          # 每个细胞错误的表征

        c = c.unsqueeze(0)
        
        return h1, h2, c

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.hgnn(seq, adj)
        c = torch.mean(h_1, dim=0)

        return h_1.detach(), c.detach()


class DGI_SAGE(nn.Module):
    def __init__(
            self,
            num_layers,
            dropout,
            device,
            in_dim: int,
            hidden_dim: int,
         ):
        super(DGI_SAGE, self).__init__()

        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim))
        self.gnn = HyperSAGE(in_dim = hidden_dim, 
                             hidden_dim = hidden_dim, 
                             out_dim = hidden_dim, 
                             num_layers = num_layers, 
                             dropout = dropout,
                             device = device)   
        self.b_xent = nn.CosineEmbeddingLoss()
    
    
    def forward(self, node_feat, neighbor_list, graph_attr):      
        feat = self.mlp(node_feat)

        nb_nodes = node_feat.shape[0]
        idx = torch.randperm(nb_nodes)
        feat_shuffled = feat[idx, :]

        h1 = self.gnn(feat, neighbor_list, graph_attr)
        h2 = self.gnn(feat_shuffled, neighbor_list, graph_attr)
        c = torch.mean(h1, dim=0).unsqueeze(0)
    
        lbl_1 = torch.ones(len(neighbor_list[0][0])).to(self.device)
        lbl_2 = -torch.ones(len(neighbor_list[0][0])).to(self.device)      
        loss = self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)
        return loss