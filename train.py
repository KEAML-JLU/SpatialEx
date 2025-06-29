import numpy as np
import torch
import torch.nn as nn
from utils import create_optimizer, create_ImageEncoder
from predictor import Predictor, Predictor_dgi, Predictor_SAGE
from hgnn import HyperSAGE, HGNN
from dgi import DGI_SAGE, DGI
import torch.nn.functional as F

class Model_vanilla(nn.Module):
    def __init__(self, args, in_dim=2048, out_dim=150):
        super(Model_vanilla, self).__init__()
        self.predictor = Predictor(
                              in_dim = in_dim,                                      # 超图训练
                              hidden_dim = args.hidden_dim,
                              out_dim = out_dim,
                              num_layers = args.num_layers,
                              loss_fn = args.loss_fn)
        
        self.dgi_model = Predictor_dgi(in_dim = in_dim,                             # dgi模型
                                  hidden_dim = args.hidden_dim,
                                  out_dim = out_dim)
    
        self.predictor.to(args.device)
        self.dgi_model.to(args.device)

    def forward(self, graph, he_rep, exp, agg_mtx, selection):
        loss_pre, x_prime, _ = self.predictor(graph, he_rep, exp, agg_mtx, selection)
        loss_dgi = self.dgi_model(graph, he_rep)
        loss = loss_pre + loss_dgi
        return loss, x_prime

    def predict(self, he_representations, H, grad=False):
        if not grad:
            with torch.no_grad():
                x_prime = self.predictor.predict(H, he_representations)
        else:
            x_prime = self.predictor.predict(H, he_representations)
        return x_prime

class Predictor_spot(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn = 'mse',
            activation = 'prelu',
            agg = True,
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
    def __init__(self, args, in_dim=2048, out_dim=150):
        super(Model, self).__init__()
        self.predictor = Predictor_spot(
                              in_dim = in_dim,                                      # 超图训练
                              hidden_dim = args.hidden_dim,
                              out_dim = out_dim,
                              num_layers = args.num_layers,
                              loss_fn = args.loss_fn)
        
        self.dgi_model = Predictor_dgi(in_dim = in_dim,                             # dgi模型
                                  hidden_dim = args.hidden_dim,
                                  out_dim = out_dim)
    
        self.predictor.to(args.device)
        self.dgi_model.to(args.device)

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
    
# class Model(nn.Module):
#     def __init__(self,                  
#                  args,
#                  adata,
#                  HE_representations,
#                  H):
#         super(Model, self).__init__()
        
#         self.predictor = Predictor(
#                               in_dim = HE_representations.shape[-1],                                      # 创建模型
#                               hidden_dim = args.hidden_dim,
#                               out_dim = adata.n_vars,
#                               num_layers = args.num_layers,
#                               loss_fn = args.loss_fn)
        
#         self.dgi_model = Predictor_dgi(in_dim = HE_representations.shape[-1],                              # 创建dgi模型
#                                   hidden_dim = args.hidden_dim,
#                                   out_dim = adata.n_vars)
        
        
#         self.predictor.to(args.device)
#         self.dgi_model.to(args.device)
        
        
#         # optimizer = create_optimizer(args.optimizer, predictor, args.lr, args.weight_decay)                       # 创建优化器
#         # optimizer_dgi = create_optimizer(args.optimizer, dgi_model, args.lr, 0)
        
#         self.HE_representations = torch.Tensor(HE_representations).to(args.device)                                  # 准备数据
        
#         #HE_copy = HE_representations.clone().detach().to(args.device)
        
#         self.exp_mtx = torch.Tensor(adata.X).to(args.device)
        
#         # 将H转换为torch.sparse.tensor
#         H = H.tocoo().astype(np.float32)
#         indices = torch.from_numpy(
#             np.vstack((H.row, H.col)).astype(np.int64))
#         values = torch.from_numpy(H.data)
#         shape = torch.Size(H.shape)
#         H = torch.sparse.FloatTensor(indices, values, shape)
#         self.H = H.to(args.device)

#     def forward(self):
#         loss_pre, x_prime, _ = self.predictor(self.H, self.HE_representations, self.exp_mtx)
#         loss_dgi = self.dgi_model(self.H, self.HE_representations)
#         loss = loss_pre+loss_dgi
#         return loss, x_prime


#     def Predict(self, args, he_representations, H):
#         he_representations = torch.Tensor(he_representations)
    
#         x_prime_list = []
    
#         H = H.tocoo().astype(np.float32)
#         indices = torch.from_numpy(
#             np.vstack((H.row, H.col)).astype(np.int64))
#         values = torch.from_numpy(H.data)
#         shape = torch.Size(H.shape)
#         H = torch.sparse.FloatTensor(indices, values, shape)

#         # perform evaluatioin
#         self.predictor.eval()
#         with torch.no_grad():
#             x_prime = self.predictor.predict(H.to(args.device), he_representations.to(args.device))

#         # do not perform evaluation
#         #x_prime = self.predictor.predict(H.to(args.device), he_representations.to(args.device))
#         return x_prime

class Model_modified(nn.Module):
    def __init__(self,    
                 args,              
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float = 0.1, 
                 activation = 'prelu',
                 use_dgi: bool = True,
                 platform: str='Xenium'):
        super(Model_modified, self).__init__()

        self.platform = platform
        self.use_dgi = use_dgi
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(hidden_dim)) 
        self.hgnn = HGNN(in_dim=hidden_dim, 
                         num_hidden=hidden_dim, 
                         out_dim=hidden_dim, 
                         num_layers=args.num_layers, 
                         dropout=dropout, 
                         activation=activation)
        self.predictor = nn.Linear(hidden_dim, out_dim)


        if self.use_dgi:
            self.dgi = DGI(hidden_dim, hidden_dim, out_dim, 'prelu')  
            self.b_xent = nn.CosineEmbeddingLoss()

        if args.loss_fn == 'mse':
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(0.1),
            )        
        self.b_xent = nn.MSELoss()
    
    def forward(self, x, y=None, agg_mtx=None):
        x = self.mlp(x)
        if y is None:
            return x
        if self.platform == 'Xenium':
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


class Model_SAGE(nn.Module):
    def __init__(self,                  
                 args,
                 hyper_graph,
                 node_feat,
                 out_dim):
        super(Model_SAGE, self).__init__()
        
        self.predictor = Predictor_SAGE(hyper_graph = hyper_graph,
                                        in_dim = node_feat.shape[-1], 
                                        hidden_dim = args.hidden_dim,
                                        out_dim = out_dim,
                                        num_layers = args.num_layers,
                                        device=args.device)
        self.predictor.to(args.device)

    def forward(self, tgt_id, node_feat, x):
        loss, x_prime, _ = self.predictor(tgt_id, node_feat, x)
        return loss, x_prime


    def predict(self, tgt_id, node_feat, hyper_graph):
        # self.predictor.eval()
        with torch.no_grad():
            x_prime = self.predictor.predict(tgt_id, node_feat, hyper_graph)
        #x_prime = self.predictor.predict(tgt_id, node_feat, hyper_graph)
        return x_prime

class Model_HP(nn.Module):
    def __init__(self,                  
                 args,
                 hyper_graph,
                 in_dim,
                 out_dim):
        super(Model_HP, self).__init__()
        
        self.use_dgi = args.use_dgi
        self.device = args.device
        self.gnn_layers = args.num_layers
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T

        self.mlp1 = nn.Sequential(nn.Linear(in_dim[0], args.hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(args.hidden_dim))  
        self.mlp2 = nn.Sequential(nn.Linear(in_dim[1], args.hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(args.hidden_dim))  
        
        self.SAGE_HA = HyperSAGE(in_dim = args.hidden_dim, 
                                 hidden_dim = args.hidden_dim, 
                                 out_dim = args.hidden_dim, 
                                 num_layers = args.num_layers, 
                                 dropout = 0.1,
                                 device = args.device)
        
        self.SAGE_HB = HyperSAGE(in_dim = args.hidden_dim, 
                                 hidden_dim = args.hidden_dim, 
                                 out_dim = args.hidden_dim, 
                                 num_layers = args.num_layers, 
                                 dropout = 0.1,
                                 device = args.device)

        self.predicter1 = nn.Linear(args.hidden_dim, out_dim[0])
        self.predicter2 = nn.Linear(args.hidden_dim, out_dim[1])

        if self.use_dgi:
            self.dgi1 = DGI_SAGE(num_layers=1,
                                dropout=0.1,
                                device=args.device,
                                in_dim=in_dim[0],
                                hidden_dim=args.hidden_dim)
            
            self.dgi2 = DGI_SAGE(num_layers=1,
                                dropout=0.1,
                                device=args.device,
                                in_dim=in_dim[1],
                                hidden_dim=args.hidden_dim)
        self.criterion = nn.MSELoss()

    def initialize_graph_attr(self, hyper_graph):
        graph_attr = {}
        graph_attr['graph'] = hyper_graph.copy()
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()                                                    # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()                                                    # 每个节点被几个超边包含
        
        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)    # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()                                               # 计算每个节点的邻居节点数量
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
            tgt_idx = np.unique(node_by_node[tgt_idx].tocoo().col)                                # 找到邻居节点
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
            loss1 = loss1 + self.dgi1(node_feat1[neighbor_list[-2][0]].to(self.device), neighbor_list[:2], self.graph_attr1)   # 完全复现之前的

        tgt_id2, node_feat2, x2, agg_mtx2  = tgt_id[1], node_feat[1], x[1], agg_mtx[1]
        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
        x_prime2 = self.predicter2(enc)
        loss2 = self.criterion(torch.spmm(agg_mtx2, x_prime2), x2)  
        if self.use_dgi:  
            loss2 = loss2 + self.dgi2(node_feat2[neighbor_list[-2][0]].to(self.device), neighbor_list[:2], self.graph_attr2) 
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
                elif which =='panelB':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                    enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                    x_prime = self.predicter2(enc)
                elif which=='both':
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
            elif which =='panelB':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                x_prime = self.predicter2(enc)
            elif which=='both':
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

    
class Model_HP_modified(nn.Module):
    def __init__(self,                  
                 args,
                 hyper_graph,
                 batch_size,
                 in_dim,
                 out_dim):
        super(Model_HP_modified, self).__init__()
        
        self.use_dgi = args.use_dgi
        self.device = args.device
        self.gnn_layers = args.num_layers
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T

        self.mlp1 = nn.Sequential(nn.Linear(in_dim[0], args.hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(args.hidden_dim))
        self.mlp2 = nn.Sequential(nn.Linear(in_dim[1], args.hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(args.hidden_dim))
        
        self.SAGE_HA = HyperSAGE(in_dim = args.hidden_dim, 
                                 hidden_dim = args.hidden_dim, 
                                 out_dim = args.hidden_dim, 
                                 num_layers = args.num_layers, 
                                 dropout = 0.1,
                                 device = args.device)
        
        self.SAGE_HB = HyperSAGE(in_dim = args.hidden_dim, 
                                 hidden_dim = args.hidden_dim, 
                                 out_dim = args.hidden_dim, 
                                 num_layers = args.num_layers, 
                                 dropout = 0.1,
                                 device = args.device)

        self.predicter1 = nn.Linear(args.hidden_dim, out_dim[0])
        self.predicter2 = nn.Linear(args.hidden_dim, out_dim[1])

        if self.use_dgi:
            self.b_xent = nn.CosineEmbeddingLoss()
            self.label1_pos = torch.ones(batch_size[0]).to(self.device)
            self.label1_neg = -torch.ones(batch_size[0]).to(self.device)
            self.label2_pos = torch.ones(batch_size[1]).to(self.device)
            self.label2_neg = -torch.ones(batch_size[1]).to(self.device)         
        self.criterion = nn.MSELoss()

    def initialize_graph_attr(self, hyper_graph):
        graph_attr = {}
        graph_attr['graph'] = hyper_graph.copy()
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()                                                    # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()                                                    # 每个节点被几个超边包含
        
        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)    # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()                                               # 计算每个节点的邻居节点数量
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
            tgt_idx = np.unique(node_by_node[tgt_idx].tocoo().col)                                # 找到邻居节点
            mapped_indices = torch.arange(tgt_idx.shape[0], device=self.device, dtype=torch.int32)
            mapping = torch.zeros(node_by_node.shape[0], dtype=torch.int32, device=self.device)
            mapping[tgt_idx] = mapped_indices
            neighbor_list.append((tgt_idx, mapping))
        return neighbor_list
    
    def forward(self, tgt_id, node_feat, x, return_prime=False):
        tgt_id1, node_feat1, x1 = tgt_id[0], node_feat[0], x[0]
        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
        enc = self.mlp1(node_feat1[neighbor_list[-1][0]])
        h1 = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
        x_prime1 = self.predicter1(h1)
        loss1 = self.criterion(x_prime1, x1)
        if self.use_dgi:
            idx = torch.randperm(enc.shape[0])
            enc_shuffled = enc[idx, :]
            h2 = self.SAGE_HA(enc_shuffled, neighbor_list, self.graph_attr1)
            c = torch.mean(h1, dim=0).unsqueeze(0)
            loss1_dgi = self.b_xent(h1, c, self.label1_pos) + self.b_xent(h2, c, self.label1_neg)
            loss1 = loss1 + loss1_dgi

        tgt_id2, node_feat2, x2 = tgt_id[1], node_feat[1], x[1]
        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
        enc = self.mlp2(node_feat2[neighbor_list[-1][0]])
        h1 = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
        x_prime2 = self.predicter2(h1)
        loss2 = self.criterion(x_prime2, x2)  
        if self.use_dgi:  
            idx = torch.randperm(enc.shape[0])
            enc_shuffled = enc[idx, :]
            h2 = self.SAGE_HB(enc_shuffled, neighbor_list, self.graph_attr2)
            c = torch.mean(h1, dim=0).unsqueeze(0)
            loss1_dgi = self.b_xent(h1, c, self.label2_pos) + self.b_xent(h2, c, self.label2_neg)
            loss2 = loss2 + loss1_dgi            
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
                elif which =='panelB':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                    enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                    x_prime = self.predicter2(enc)
                elif which=='both':
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
            elif which =='panelB':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                x_prime = self.predicter2(enc)
            elif which=='both':
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
        
class Model_uniSAGE(nn.Module):
    def __init__(self,                  
                 args,
                 adata,
                 HE_representations,
                 H):
        super(Model, self).__init__()
        
        self.predictor = Predictor_uniSAGE(
                              in_dim = HE_representations.shape[-1],                                      # 创建模型
                              hidden_dim = args.hidden_dim,
                              out_dim = adata.n_vars,
                              num_layers = args.num_layers,
                              loss_fn = args.loss_fn)
        
        self.dgi_model = Predictor_dgi(in_dim = HE_representations.shape[-1],                              # 创建dgi模型
                                  hidden_dim = args.hidden_dim,
                                  out_dim = adata.n_vars)
        
        
        self.predictor.to(args.device)
        self.dgi_model.to(args.device)
        
        self.HE_representations = torch.Tensor(HE_representations).to(args.device)                                  # 准备数据
        
        self.exp_mtx = torch.Tensor(adata.X).to(args.device)
        
        # 将H转换为torch.sparse.tensor
        H = H.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((H.row, H.col)).astype(np.int64))
        values = torch.from_numpy(H.data)
        shape = torch.Size(H.shape)
        H = torch.sparse.FloatTensor(indices, values, shape)
        self.H = H.to(args.device)

    def forward(self):
        loss_pre, x_prime, _ = self.predictor(self.H, self.HE_representations, self.exp_mtx)
        loss_dgi = self.dgi_model(self.H, self.HE_representations)
        loss = loss_pre+loss_dgi
        return loss, x_prime


    def Predict(self, args, he_representations, H):
        he_representations = torch.Tensor(he_representations)
    
        x_prime_list = []
    
        H = H.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((H.row, H.col)).astype(np.int64))
        values = torch.from_numpy(H.data)
        shape = torch.Size(H.shape)
        H = torch.sparse.FloatTensor(indices, values, shape)

        # perform evaluatioin
        self.predictor.eval()
        with torch.no_grad():
            x_prime = self.predictor.predict(H.to(args.device), he_representations.to(args.device))

        return x_prime


class Model_HP_Singleomics(nn.Module):
    def __init__(self,
                 args,
                 hyper_graph,
                 in_dim,
                 out_dim):
        super(Model_HP_Singleomics, self).__init__()

        self.use_dgi = args.use_dgi
        self.device = args.device
        self.gnn_layers = args.num_layers
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph)
        self.node_by_node1 = hyper_graph @ hyper_graph.T

        self.mlp1 = nn.Sequential(nn.Linear(in_dim, args.hidden_dim),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(args.hidden_dim))

        self.SAGE_HA = HyperSAGE(in_dim=args.hidden_dim,
                                 hidden_dim=args.hidden_dim,
                                 out_dim=args.hidden_dim,
                                 num_layers=args.num_layers,
                                 dropout=0.1,
                                 device=args.device)

        self.predicter1 = nn.Linear(args.hidden_dim, out_dim)

        if self.use_dgi:
            self.dgi1 = DGI_SAGE(num_layers=1,
                                 dropout=0.1,
                                 device=args.device,
                                 in_dim=in_dim,
                                 hidden_dim=args.hidden_dim)

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

    def forward(self, tgt_id, node_feat, x, agg_mtx, return_prime=False):
        tgt_id1, node_feat1, x1, agg_mtx1 = tgt_id, node_feat, x, agg_mtx
        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
        enc = self.mlp1(node_feat1[neighbor_list[-1][0]])
        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
        x_prime1 = self.predicter1(enc)
        loss1 = self.criterion(torch.spmm(agg_mtx1, x_prime1), x1)
        if self.use_dgi:
            loss1 = loss1 + self.dgi1(node_feat1[neighbor_list[-2][0]], neighbor_list[:2], self.graph_attr1)  # 完全复现之前的

        if return_prime:
            return loss1, x_prime1

        return loss1

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
                    if exchange:
                        x_prime = []
                        tgt_id1, node_feat1 = tgt_id, node_feat
                        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                        x_prime.append(self.predicter1(enc))

                    # if not exchange:
                    #     x_prime = []
                    #     tgt_id1, node_feat1 = tgt_id, node_feat
                    #     neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                    #     enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    #     enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                    #     x_prime.append(self.predicter1(enc))

                    # else:
                    #     x_prime = []
                    #     tgt_id1, node_feat1 = tgt_id[1], node_feat[1]
                    #     neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node2)
                    #     enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    #     enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr2)
                    #     x_prime.append(self.predicter1(enc))
                    #
                    #     tgt_id2, node_feat2 = tgt_id[0], node_feat[0]
                    #     neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node1)
                    #     enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                    #     enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr1)
                    #     x_prime.append(self.predicter2(enc))
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

class EarlyStopping:
    def __init__(self, patience=100, delta=0.01):
        self.patience= patience
        self.max_acc = -1
        self.counter = 0
        self.delta = delta
        self.early_stop = False


    def __call__(self, val_acc):
        if self.max_acc == -1:
            self.max_acc = val_acc
        elif val_acc < self.max_acc-self.delta:                   # 如果准确率连续降低超过10次，那么提前停止
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        else:   
            self.counter = 0
            if val_acc>self.max_acc:                                                  
                self.max_acc = val_acc 
        return self.early_stop


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=False, device='cpu'):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            num_classes = len(alpha)
            self.alpha = torch.Tensor(alpha).to(device)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1,preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss