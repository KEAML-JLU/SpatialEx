import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_activation


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
            for i in range(self.num_layers - 2):
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
            for i in range(self.num_layers - 2):
                X = torch.sparse.mm(H, self.mlp[i](self.dropout(X)))
                X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))

        return X


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
        h1 = self.hgnn(seq1, adj)  # 每个细胞正确的表征

        c = torch.mean(h1, dim=0)  # 正确的全局表征

        h2 = self.hgnn(seq2, adj)  # 每个细胞错误的表征

        c = c.unsqueeze(0)

        return h1, h2, c

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.hgnn(seq, adj)
        c = torch.mean(h_1, dim=0)

        return h_1.detach(), c.detach()


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
            self.weight_list = [nn.Parameter(torch.FloatTensor(2 * in_dim, hidden_dim)).to(self.device), ]
            for _ in range(self.num_layers - 2):
                self.weight_list.append(nn.Parameter(torch.FloatTensor(2 * hidden_dim, hidden_dim)).to(self.device))
            self.weight_list.append(nn.Parameter(torch.FloatTensor(2 * hidden_dim, out_dim)).to(self.device))
        else:
            self.weight_list = [nn.Parameter(torch.FloatTensor(2 * in_dim, out_dim)).to(self.device), ]
        for weight in self.weight_list:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, node_feat, neighbor_list, graph_attr):
        '''训练过程中会提前计算好一些图属性以加速训练'''
        for layer in range(self.num_layers):
            neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - layer - 1][0],
                                                        neighbor_list[self.num_layers - layer][0], node_feat,
                                                        graph_attr)
            map_dict = neighbor_list[self.num_layers - layer][1]
            tgt_index = map_dict[neighbor_list[self.num_layers - 1 - layer][0]]  # 全局索引映射回上一层索引
            feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
            node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight_list[layer]))
        return node_feat

    def predict(self, node_feat, neighbor_list, graph_attr):
        '''需要重新计算超图相关属性'''
        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - 1][0],
                                                    neighbor_list[self.num_layers][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers][1]
        tgt_index = map_dict[neighbor_list[self.num_layers - 1][0]]
        feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
        node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight1))

        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - 2][0],
                                                    neighbor_list[self.num_layers - 1][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers - 1][1]
        tgt_index = map_dict[neighbor_list[self.num_layers - 2][0]]
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
            return torch.sparse.FloatTensor(indices, values, shape)

    def sparse_diags(self, data):
        indices = torch.arange(data.shape[0])
        indices = torch.vstack([indices, indices]).to(self.device)
        return torch.sparse_coo_tensor(indices, data, (data.shape[0], data.shape[0]))

    def initialize(self, hyper_graph):
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()  # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()  # 每个节点被几个超边包含

        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)  # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()  # 计算每个节点的邻居节点数量
        return num_nodes, num_edges, num_neighbors

    def Aggregate_neighbors(self, tgt_idx, src_idx, node_emb, graph_attr=None):
        '''
        前向传播每次需要聚合邻居节点
        tgt_idx,            [list],             本层需要更新的目标节点
        src_idx,            [list],             本层目标节点的邻居节点
        node_emb,           [torch.Tensor, 2D], 前一层的节点表征
        model,              [str]             ,
        '''

        hyper_graph, num_nodes, num_edges, num_neighbors = graph_attr['graph'], graph_attr['num_nodes'], graph_attr[
            'num_edges'], graph_attr['num_neighbors']

        '''将源节点特征聚合到超边上'''
        tgt_edge = np.unique(hyper_graph[tgt_idx].tocoo().col)  # 相关超边，只有稀疏array支持索引
        edge_cardinality = num_nodes[tgt_edge]
        edge_cardinality_inv = 1.0 / edge_cardinality
        edge_cardinality = self.sparse_diags(edge_cardinality)
        edge_cardinality_inv = self.sparse_diags(edge_cardinality_inv)

        edge_agg_mtx = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[src_idx][:, tgt_edge].T,
                                                             cuda=True)  # 目标节点参与的超边*邻接节点的聚合矩阵
        edge_emb = torch.spmm(torch.spmm(edge_cardinality_inv, edge_agg_mtx), node_emb)  # 将节点特征聚合到超边上

        '''将超边特征聚合到目标节点上'''
        num_neighbor_inv = 1.0 / num_neighbors[tgt_idx]
        num_neighbor_inv = self.sparse_diags(num_neighbor_inv)  # 算目标节点所有通过超边邻接的节点数量
        num_edge_inv = 1.0 / num_edges[tgt_idx]
        num_edge_inv = self.sparse_diags(num_edge_inv)

        tgt_by_edge = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[tgt_idx][:, tgt_edge], cuda=True)
        neighbor_agg_emb = torch.spmm(num_edge_inv,
                                      num_neighbor_inv @ torch.spmm(torch.spmm(tgt_by_edge, edge_cardinality),
                                                                    edge_emb))
        return neighbor_agg_emb


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
            nn.BatchNorm1d(hidden_dim)).to(device)
        self.gnn = HyperSAGE(in_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             out_dim=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             device=device)
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


class Predictor_spot(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn='mse',
            activation='prelu',
            agg=True,
    ):
        super(Predictor_spot, self).__init__()

        dropout = 0
        self.agg = agg
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)
        )

        self.mod = HGNN(
            in_dim=hidden_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )

        self.linear = nn.Linear(hidden_dim, out_dim)

        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        if self.agg:
            loss = self.criterion(torch.sparse.mm(agg_mtx, x_prime[selection]), x)
        else:
            loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        return x_prime


class Model(nn.Module):
    def __init__(self,
                 num_layers=2,
                 in_dim=2048,
                 hidden_dim=512,
                 out_dim=150,
                 loss_fn="mse",
                 device="cpu"
                 ):
        super(Model, self).__init__()
        self.predictor = Predictor_spot(
            in_dim=in_dim,  # 超图训练
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            loss_fn=loss_fn)

        self.dgi_model = Predictor_dgi(in_dim=in_dim,  # dgi模型
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim)

        self.predictor.to(device)
        self.dgi_model.to(device)

    def forward(self, graph, he_rep, exp, agg_mtx, selection):
        loss_pre, x_prime, _ = self.predictor(graph, he_rep, exp, agg_mtx, selection)
        loss_dgi = self.dgi_model(graph, he_rep)
        loss = loss_pre + loss_dgi
        return loss, x_prime

    def predict(self, he_representations, graph, grad=False):
        if not grad:
            with torch.no_grad():
                x_prime = self.predictor.predict(graph, he_representations)
        else:
            x_prime = self.predictor.predict(graph, he_representations)
        return x_prime


class Model_Plus(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 activation='prelu',
                 use_dgi: bool = True,
                 loss_fn: str = 'mse',
                 platform: str = 'Xenium'):
        super(Model_Plus, self).__init__()

        self.platform = platform
        self.use_dgi = use_dgi
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(hidden_dim))
        self.hgnn = HGNN(in_dim=hidden_dim,
                         num_hidden=hidden_dim,
                         out_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         activation=activation)
        self.predictor = nn.Linear(hidden_dim, out_dim)

        if self.use_dgi:
            self.dgi = DGI(hidden_dim, hidden_dim, out_dim, 'prelu')
            self.b_xent = nn.CosineEmbeddingLoss()

        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()

    def forward(self, x, adj, y, agg_mtx=None):
        x = self.mlp(x)
        h = F.leaky_relu(self.hgnn(x, adj))
        x_prime = F.leaky_relu(self.predictor(h))
        if self.platform == 'Visium':
            loss = self.criterion(x_prime, y)
        else:
            loss = self.criterion(torch.mm(agg_mtx, x_prime), y)
        if self.use_dgi:
            nb_nodes = x.shape[0]
            x_shuffle = x[torch.randperm(nb_nodes)]
            h1, h2, c = self.dgi(x, x_shuffle, adj)
            lbl_1 = torch.ones(nb_nodes).to(x.device)
            lbl_2 = -torch.ones(nb_nodes).to(x.device)
            loss = loss + self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)
        return loss, x_prime

    def predict(self, x, adj, grad=False):
        if not grad:
            with torch.no_grad():
                x = self.mlp(x)
                h = F.leaky_relu(self.hgnn(x, adj))
                x_prime = F.leaky_relu(self.predictor(h))
        else:
            x = self.mlp(x)
            h = F.leaky_relu(self.hgnn(x, adj))
            x_prime = F.leaky_relu(self.predictor(h))
        return x_prime


class Regression(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            platform: str = 'Xenium',
    ):
        super(Regression, self).__init__()

        self.platform = platform
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(0.1),
        )
        self.b_xent = nn.MSELoss()

    def forward(self, x, y=None, agg_mtx=None):
        x = self.mlp(x)
        if y is None:
            return x
        if agg_mtx != None:
            loss = self.b_xent(torch.spmm(agg_mtx, x), y)
        else:
            loss = self.b_xent(x, y)
        return loss, x

    def predict(self, x, grad=False):
        if not grad:
            with torch.no_grad():
                x = self.mlp(x)
        else:
            x = self.mlp(x)
        return x


class Classifier(torch.nn.Module):
    def __init__(self,
                 dim_input,
                 dim_hidden,
                 dim_output,
                 alpha,
                 device):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(dim_input, dim_hidden)
        self.hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.hidden3 = nn.Linear(dim_hidden, dim_output)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.criterion = focal_loss(alpha=alpha, device=device)

    def forward(self, x, y):
        x = self.add_gaussian_noise(x)
        h = F.leaky_relu(F.dropout(self.hidden1(x)))
        h = F.leaky_relu(F.dropout(self.hidden2(h)))
        h = F.leaky_relu(self.hidden3(h))
        loss = self.criterion(h, y)
        return loss

    def predict(self, x):
        h = F.leaky_relu(F.dropout(self.hidden1(x)))
        h = F.leaky_relu(F.dropout(self.hidden2(h)))
        h = F.leaky_relu(self.hidden3(h))
        return h

    def add_gaussian_noise(self, x, mean=0, std=0.1):
        noise = torch.randn_like(x) * std + mean
        return x + noise.to(x.device)


class Model_Big(nn.Module):
    def __init__(self,
                 hyper_graph,
                 in_dim,
                 out_dim,
                 num_layers,
                 hidden_dim,
                 device,
                 use_dgi=True):
        super(Model_Big, self).__init__()

        self.use_dgi = use_dgi
        self.device = device
        self.gnn_layers = num_layers
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T

        self.mlp1 = nn.Sequential(nn.Linear(in_dim[0], hidden_dim),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(hidden_dim))
        self.mlp2 = nn.Sequential(nn.Linear(in_dim[1], hidden_dim),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(hidden_dim))

        self.SAGE_HA = HyperSAGE(in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_layers=num_layers,
                                 dropout=0.1,
                                 device=device)

        self.SAGE_HB = HyperSAGE(in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_layers=num_layers,
                                 dropout=0.1,
                                 device=device)

        self.predicter1 = nn.Linear(hidden_dim, out_dim[0])
        self.predicter2 = nn.Linear(hidden_dim, out_dim[1])

        if self.use_dgi:
            self.dgi1 = DGI_SAGE(num_layers=1,
                                 dropout=0.1,
                                 device=device,
                                 in_dim=in_dim[0],
                                 hidden_dim=hidden_dim)

            self.dgi2 = DGI_SAGE(num_layers=1,
                                 dropout=0.1,
                                 device=device,
                                 in_dim=in_dim[1],
                                 hidden_dim=hidden_dim)
        self.criterion = nn.MSELoss()

    def initialize_graph_attr(self, hyper_graph):
        graph_attr = {}
        graph_attr['graph'] = hyper_graph.copy()
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()  # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()  # 每个节点被几个超边包含

        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)  # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()  # 计算每个节点的邻居节点数量
        graph_attr['num_nodes'] = num_nodes
        graph_attr['num_edges'] = num_edges
        graph_attr['num_neighbors'] = num_neighbors
        return graph_attr

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx, cuda=False):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        if cuda:
            return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        else:
            return torch.sparse.FloatTensor(indices, values, shape)

    def get_neighbors(self, tgt_idx, node_by_node):
        neighbor_list = [(tgt_idx, None), ]
        for _ in range(self.gnn_layers):
            tgt_idx = np.unique(node_by_node[tgt_idx].tocoo().col)  # 找到邻居节点
            mapped_indices = torch.arange(tgt_idx.shape[0], device=self.device, dtype=torch.int32)
            mapping = torch.zeros(node_by_node.shape[0], dtype=torch.int32, device=self.device)
            mapping[tgt_idx] = mapped_indices
            neighbor_list.append((tgt_idx, mapping))
        return neighbor_list

    def forward(self, tgt_id, node_feat, x, agg_mtx=None, return_prime=False):
        tgt_id1, node_feat1, x1, agg_mtx1 = tgt_id[0], node_feat[0], x[0], agg_mtx[0]
        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
        x_prime1 = self.predicter1(enc)
        loss1 = self.criterion(torch.spmm(agg_mtx1, x_prime1), x1)
        if self.use_dgi:
            loss1 = loss1 + self.dgi1(node_feat1[neighbor_list[-2][0]].to(self.device), neighbor_list[:2],
                                      self.graph_attr1)  # 完全复现之前的

        tgt_id2, node_feat2, x2, agg_mtx2 = tgt_id[1], node_feat[1], x[1], agg_mtx[1]
        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
        x_prime2 = self.predicter2(enc)
        loss2 = self.criterion(torch.spmm(agg_mtx2, x_prime2), x2)
        if self.use_dgi:
            loss2 = loss2 + self.dgi2(node_feat2[neighbor_list[-2][0]].to(self.device), neighbor_list[:2],
                                      self.graph_attr2)
        if return_prime:
            return loss1, loss2, x_prime1, x_prime2

        return loss1, loss2

    def predict(self, tgt_id, node_feat, exchange=False, which='both', grad=False):
        if not grad:
            with torch.no_grad():
                if which == 'panelA':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node1)
                    enc = self.mlp1(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(node_feat, neighbor_list, self.graph_attr1)
                    x_prime = self.predicter1(enc)
                elif which == 'panelB':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                    enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                    x_prime = self.predicter2(enc)
                elif which == 'both':
                    if not exchange:
                        x_prime = []
                        tgt_id1, node_feat1 = tgt_id[0], node_feat[0]
                        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                        x_prime.append(self.predicter1(enc))

                        tgt_id2, node_feat2 = tgt_id[1], node_feat[1]
                        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
                        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
                        x_prime.append(self.predicter2(enc))
                    else:
                        x_prime = []
                        tgt_id1, node_feat1 = tgt_id[1], node_feat[1]
                        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node2)
                        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr2)
                        x_prime.append(self.predicter1(enc))

                        tgt_id2, node_feat2 = tgt_id[0], node_feat[0]
                        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node1)
                        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr1)
                        x_prime.append(self.predicter2(enc))
                else:
                    print('Please specify the panel you want to predict: panelA/panelB/both.')
        else:
            if which == 'panelA':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node1)
                enc = self.mlp1(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HA(node_feat, neighbor_list, self.graph_attr1)
                x_prime = self.predicter1(enc)
            elif which == 'panelB':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                x_prime = self.predicter2(enc)
            elif which == 'both':
                if not exchange:
                    x_prime = []
                    tgt_id1, node_feat1 = tgt_id[0], node_feat[0]
                    neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                    enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                    x_prime.append(self.predicter1(enc))

                    tgt_id2, node_feat2 = tgt_id[1], node_feat[1]
                    neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
                    enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
                    x_prime.append(self.predicter2(enc))
                else:
                    x_prime = []
                    tgt_id1, node_feat1 = tgt_id[1], node_feat[1]
                    neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node2)
                    enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr2)
                    x_prime.append(self.predicter1(enc))

                    tgt_id2, node_feat2 = tgt_id[0], node_feat[0]
                    neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node1)
                    enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr1)
                    x_prime.append(self.predicter2(enc))
            else:
                print('Please specify the panel you want to predict: panelA/panelB/both.')
        return x_prime

    def set_graph_attr(self, hyper_graph):
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=False, device='cpu'):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            num_classes = len(alpha)
            self.alpha = torch.Tensor(alpha).to(device)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Predictor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn='mse',
            activation='prelu',
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