"""
SpatialEx training utilities.

This module contains three trainer classes:

- :class:`Train_SpatialEx`: Trains two SpatialEx models (one per slice) and
  evaluates cross-panel prediction quality via cosine similarity, SSIM, PCC,
  and CMD.
- :class:`Train_SpatialExP`: Trains SpatialEx+ with additional regression
  mapping heads in a cycle-style setup to translate between gene panels.
- :class:`Train_SpatialExP_Big`: Trains SpatialEx+ specifically for millons cells.

The trainers expect two AnnData slices whose `.obsm['he']` stores
histology-derived embeddings and whose `var_names` align with gene features.

Note:
    This module imports project-specific components from sibling modules:
    ``model.Model``, ``model.Model_Plus``, ``model.Model_Big``, ``model.Regression``,
    ``utils.create_optimizer``, ``utils.Compute_metrics``, and preprocessing functions as ``pp``.
"""

import os
import torch
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from . import preprocess as pp
from .utils import create_optimizer, Generate_pseudo_spot
from .model import Model, Model_Plus, Model_Big, Regression


warnings.filterwarnings("ignore")


class Train_SpatialEx:
    """Trainer for baseline SpatialEx on two slices.

    This trainer fits two models (:attr:`module_HA` for slice 1 and
    :attr:`module_HB` for slice 2) independently using hypergraph-based
    batches, then evaluates cross-panel predictions at the end.

    Attributes:

    adata1 (AnnData): Slice 1.

    adata2 (AnnData): Slice 2.

    num_layers (int): Number of HGNN layers.

    hidden_dim (int): Hidden width of the backbone.

    epochs (int): Number of training epochs.

    seed (int): Random seed.

    device (torch.device): Device on which models are trained.

    weight_decay (float): Weight decay for the optimizer.

    optimizer (torch.optim.Optimizer): Optimizer instance.

    batch_size (int): Batch size when building the hypergraph.

    encoder (str): Encoder architecture key (e.g., ``"hgnn"``).

    lr (float): Learning rate.

    loss_fn (str): Loss function key (e.g., ``"mse"``).

    num_neighbors (int): K for KNN used in hypergraph construction.

    graph_kind (str): Spatial graph/hypergraph type (e.g., ``"spatial"``).

    prune (int): Pruning threshold for dataloader construction.

    save (bool): Whether to save the results.

    """


    def __init__(self,
                 adata1,
                 adata2,
                 num_layers=2,
                 hidden_dim=512,
                 epochs=500,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 encoder="hgnn",
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 prune=10000,
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.encoder = encoder

        self.lr = lr
        self.loss_fn = loss_fn
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.prune = prune
        self.save = save

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        self.module_HA = Model(self.num_layers, self.in_dim1, self.hidden_dim, self.out_dim1, self.loss_fn, self.device)
        self.module_HB = Model(self.num_layers, self.in_dim2, self.hidden_dim, self.out_dim2, self.loss_fn, self.device)
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.slice1_dataloader = pp.Build_dataloader(adata1, graph=H1, graph_norm='hpnn', feat_norm=False,
                                                     prune=[prune, prune], drop_last=False)
        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.slice2_dataloader = pp.Build_dataloader(adata2, graph=H2, graph_norm='hpnn', feat_norm=False,
                                                     prune=[prune, prune], drop_last=False)

    def train(self):
        """Run the training loop and evaluate cross-panel predictions.

        The method trains :attr:`module_HA` and :attr:`module_HB` jointly by
        iterating over paired mini-batches from two slices. After training, it
        predicts the missing panel on each slice and computes metrics at
        gene-level (cosine similarity, SSIM, PCC, CMD).

        Args:
            data_dir: Project root containing a ``datasets/`` folder with:
                - ``Human_Breast_Cancer_Rep1/cell_feature_matrix.h5``
                - ``Human_Breast_Cancer_Rep1/cells.csv``
                - ``Human_Breast_Cancer_Rep2/cell_feature_matrix.h5``
                - ``Human_Breast_Cancer_Rep2/cells.csv``

        Prints:
            Aggregated metrics per slice (cosine similarity, SSIM, PCC, CMD).

        Raises:
            FileNotFoundError: If any expected dataset file is missing.

        Returns:
            None
        """
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        print('\n')
        print('=================================== Start training =========================================')
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            batch_iter = zip(self.slice1_dataloader, self.slice2_dataloader)
            for data1, data2 in batch_iter:
                graph1, he1, panel_1a, selection1 = data1[0]['graph'].to(self.device), data1[0]['he'].to(self.device), \
                    data1[0]['exp'].to(self.device), data1[0]['selection']
                graph2, he2, panel_2b, selection2 = data2[0]['graph'].to(self.device), data2[0]['he'].to(self.device), \
                    data2[0]['exp'].to(self.device), data2[0]['selection']
                agg_mtx1, agg_exp1 = data1[0]['agg_mtx'].to(self.device), data1[0]['agg_exp'].to(self.device)
                agg_mtx2, agg_exp2 = data2[0]['agg_mtx'].to(self.device), data2[0]['agg_exp'].to(self.device)

                loss1, _ = self.module_HA(graph1, he1, agg_exp1, agg_mtx1, selection1)
                loss2, _ = self.module_HB(graph2, he2, agg_exp2, agg_mtx2, selection2)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_iter.set_description(f"#Epoch: {epoch}: train_loss: {loss.item():.2f}")

        '''========================= 测试 ========================'''
        self.module_HA.eval()
        self.module_HB.eval()
        '''PanelA1'''
        panel_1b = []
        obs_list = []
        for data in self.slice1_dataloader:
            graph, he, obs = data[0]['graph'].to(self.device), data[0]['he'].to(self.device), data[0]['obs']
            panelB1 = self.module_HB.predict(he, graph).detach().cpu().numpy()
            panel_1b.append(panelB1)
            obs_list = obs_list + obs
        panel_1b = np.vstack(panel_1b)
        panel_1b = pd.DataFrame(panel_1b)
        panel_1b.columns = self.adata1.var_names
        panel_1b['obs_name'] = obs_list
        panel_1b = panel_1b.groupby('obs_name').mean()

        '''Panel2B'''
        panel_2a = []
        obs_list = []
        for data in self.slice2_dataloader:
            graph, he, obs = data[0]['graph'].to(self.device), data[0]['he'].to(self.device), data[0]['obs']
            panel2A = self.module_HA.predict(he, graph).detach().cpu().numpy()
            panel_2a.append(panel2A)
            obs_list = obs_list + obs
        panel_2a = np.vstack(panel_2a)
        panel_2a = pd.DataFrame(panel_2a)
        panel_2a.columns = self.adata2.var_names
        panel_2a['obs_name'] = obs_list
        panel_2a = panel_2a.groupby('obs_name').mean()

        if self.save:
            save_path = './results/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            panel_1b.to_csv(save_path + 'HE_to_omics_panel1b.csv')
            panel_2a.to_csv(save_path + 'HE_to_omics_panel2a.csv')
            print(f'The results have been sucessfully saved in {save_path}')


class Train_SpatialExP:
    def __init__(self,
                 adata1,
                 adata2,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 encoder="hgnn",
                 hidden_dim=512,
                 num_layers=2,
                 epochs=1000,
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        # 基础参数
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.save = save

        # 空间参数
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.H1 = pp.sparse_mx_to_torch_sparse_tensor(H1).to(self.device)
        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.H2 = pp.sparse_mx_to_torch_sparse_tensor(H2).to(self.device)

        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']).to(self.device), torch.Tensor(adata2.obsm['he']).to(self.device)
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X).to(self.device), torch.Tensor(adata2.X).to(self.device)

        self.in_dim1 = adata1.obsm['he'].shape[1]
        self.in_dim2 = adata2.obsm['he'].shape[1]
        self.out_dim1 = adata1.n_vars
        self.out_dim2 = adata2.n_vars

        self.module_HA = Model_Plus(in_dim=self.in_dim1, hidden_dim=self.hidden_dim, out_dim=self.out_dim1, num_layers=self.num_layers,
                                   platform='Visium').to(self.device)
        self.module_HB = Model_Plus(in_dim=self.in_dim2, hidden_dim=self.hidden_dim, out_dim=self.out_dim2, num_layers=self.num_layers,
                                   platform='Visium').to(self.device)

        self.rm_AB = Regression(self.out_dim1, self.out_dim2, self.out_dim2).to(self.device)
        self.rm_BA = Regression(self.out_dim2, self.out_dim1, self.out_dim1).to(self.device)
        self.models = [self.module_HA, self.module_HB, self.rm_AB, self.rm_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

    def train(self):
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        self.rm_AB.train()
        self.rm_BA.train()
        print('\n')
        print('=================================== Start training =========================================')
        for epoch in tqdm(range(self.epochs)):
            loss1, _ = self.module_HA(self.HE1, self.H1, self.panelA1)
            loss2, _ = self.module_HB(self.HE2, self.H2, self.panelB2)

            panelA2 = self.module_HA.predict(self.HE2, self.H2, grad=False)
            panelB1 = self.module_HB.predict(self.HE1, self.H1, grad=False)
            loss3, _ = self.rm_AB(panelA2, self.panelB2)
            loss4, _ = self.rm_BA(panelB1, self.panelA1)

            loss5, _ = self.rm_AB(self.panelA1, panelB1)
            loss6, _ = self.rm_BA(self.panelB2, panelA2)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        '''========================= 测试 ========================'''
        self.module_HA.eval()
        self.module_HB.eval()
        self.rm_AB.eval()
        self.rm_BA.eval()

        '''PanelB1'''
        panelA1_direct = self.module_HA.predict(self.HE1, self.H1, grad=False)
        omics1_indirect = self.rm_AB.predict(panelA1_direct)


        '''PanelA2'''
        panelB2_direct = self.module_HB.predict(self.HE2, self.H2, grad=False)
        omics2_indirect = self.rm_BA.predict(panelB2_direct)

        omics1_indirect = omics1_indirect.detach().cpu().numpy()
        omics2_indirect = omics2_indirect.detach().cpu().numpy()

        if self.save:
            save_path = './results/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.save(save_path + 'omics1.npy', omics1_indirect)
            np.save(save_path + 'omics2.npy', omics2_indirect)
            print(f'The results have been sucessfully saved in {save_path}')


class Train_SpatialExP_Big:
    def __init__(self,
                 adata1,
                 adata2,
                 num_layers=2,
                 hidden_dim=512,
                 epochs=500,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 batch_num=10,
                 encoder="hgnn",
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.batch_num = batch_num
        self.encoder = encoder

        self.lr = lr
        self.loss_fn = loss_fn
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.save = save

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata1 = Generate_pseudo_spot(adata1, all_in=True)
        spot_id = adata1.obs['spot'].values
        head = spot_id[~pd.isna(adata1.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata1.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx1 = sp.coo_matrix((values, (head, tail)), shape=(head.max() + 1, adata1.n_obs)).tocsr()
        self.spot_A1 = torch.Tensor(self.agg_mtx1 @ adata1.X)

        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata2 = Generate_pseudo_spot(adata2, all_in=True)
        spot_id = adata2.obs['spot'].values
        head = spot_id[~pd.isna(adata2.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata2.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx2 = sp.coo_matrix((values, (head, tail)), shape=(head.max()+1, adata2.n_obs)).tocsr()
        self.spot_B2 = torch.Tensor(self.agg_mtx2 @ adata2.X)

        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']), torch.Tensor(adata2.obsm['he'])
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X), torch.Tensor(adata2.X)

        self.model_big = Model_Big([H1, H2], [self.in_dim1, self.in_dim2], [self.out_dim1, self.out_dim2], num_layers=self.num_layers,
                                   hidden_dim=self.hidden_dim, device=self.device).to(self.device)
        self.model_AB = Regression(self.out_dim1, int(self.out_dim1/2), self.out_dim2).to(self.device)
        self.model_BA = Regression(self.out_dim2, int(self.out_dim1/2), self.out_dim1).to(self.device)
        self.models = [self.model_big, self.model_AB, self.model_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

    def train(self):
        batch_num = self.batch_num
        obs_index1 = list(range(self.agg_mtx1.shape[0]))
        obs_index2 = list(range(self.agg_mtx2.shape[0]))
        batch_size1 = int(self.agg_mtx1.shape[0]/batch_num)
        batch_size2 = int(self.agg_mtx2.shape[0]/batch_num)
        for epoch in range(self.epochs):
            random.shuffle(obs_index1)
            random.shuffle(obs_index2)
            batch_iter = tqdm(range(batch_num), leave=False)
            for batch_idx in batch_iter:
                torch.cuda.empty_cache()
                tgt_spot1 = obs_index1[batch_idx*batch_size1:(batch_idx+1)*batch_size1]
                tgt_cell1 = self.agg_mtx1[tgt_spot1].tocoo().col
                sub_agg_mtx1 = self.agg_mtx1[tgt_spot1][:,tgt_cell1]
                sub_agg_mtx1 = pp.sparse_mx_to_torch_sparse_tensor(sub_agg_mtx1).to(self.device)
                spot_A1_batch = self.spot_A1[tgt_spot1].to(self.device)

                tgt_spot2 = obs_index2[batch_idx*batch_size2:(batch_idx+1)*batch_size2]
                tgt_cell2 = self.agg_mtx2[tgt_spot2].tocoo().col
                sub_agg_mtx2 = self.agg_mtx2[tgt_spot2][:,tgt_cell2]
                sub_agg_mtx2 = pp.sparse_mx_to_torch_sparse_tensor(sub_agg_mtx2).to(self.device)
                spot_B2_batch = self.spot_B2[tgt_spot2].to(self.device)

                loss1, loss2 = self.model_big([tgt_cell1, tgt_cell2], [self.HE1, self.HE2], [spot_A1_batch, spot_B2_batch], [sub_agg_mtx1, sub_agg_mtx2])

                x_prime = self.model_big.predict([tgt_cell1, tgt_cell2], [self.HE1, self.HE2], exchange=True, which='both', grad=False)
                panel_A2, panel_B1 = x_prime[0], x_prime[1]
                loss3, _ = self.model_AB(panel_A2, spot_B2_batch, sub_agg_mtx2)
                loss4, _ = self.model_BA(panel_B1, spot_A1_batch, sub_agg_mtx1)
                loss5, _ = self.model_AB(self.panelA1[tgt_cell1].to(self.device), torch.spmm(sub_agg_mtx1, panel_B1), sub_agg_mtx1)
                loss6, _ = self.model_BA(self.panelB2[tgt_cell2].to(self.device), torch.spmm(sub_agg_mtx2, panel_A2), sub_agg_mtx2)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_iter.set_description(
                f"#Epoch {epoch}, loss1: {round(loss1.item(), 2)}, loss2: {round(loss2.item(), 2)}, loss3: {round(loss3.item(), 2)}, loss4: {round(loss4.item(), 2)}, loss5: {round(loss5.item(), 2)}, loss6: {round(loss6.item(), 2)}")

        '''测试'''
        obs_index1 = list(range(self.HE1.shape[0]))
        obs_index2 = list(range(self.HE2.shape[0]))
        batch_size1 = int(np.ceil(self.HE1.shape[0]/batch_num))
        batch_size2 = int(np.ceil(self.HE2.shape[0]/batch_num))
        batch_iter = tqdm(range(batch_num), leave=False)

        indirect_panel_B1_list = []
        indirect_panel_A2_list = []
        tgt_id1_list = []
        tgt_id2_list = []
        self.model_big.eval()
        self.model_AB.eval()
        self.model_BA.eval()
        for batch_idx in batch_iter:
            tgt_id1 = obs_index1[batch_idx*batch_size1:min((batch_idx+1)*batch_size1, self.HE1.shape[0])]
            tgt_id2 = obs_index2[batch_idx*batch_size2:min((batch_idx+1)*batch_size2, self.HE2.shape[0])]

            x_prime = self.model_big.predict([tgt_id1, tgt_id2], [self.HE1, self.HE2], exchange=False, which='both')
            panel_A1_predict, panel_B2_predict = x_prime[0], x_prime[1]

            indirect_panel_B1 = self.model_AB.predict(panel_A1_predict)
            indirect_panel_A2 = self.model_BA.predict(panel_B2_predict)

            tgt_id1_list = tgt_id1_list + tgt_id1
            tgt_id2_list = tgt_id2_list + tgt_id2

            indirect_panel_A2_list.append(indirect_panel_A2.detach().cpu().numpy())
            indirect_panel_B1_list.append(indirect_panel_B1.detach().cpu().numpy())


        indirect_panel_A2_list = np.vstack(indirect_panel_A2_list)
        indirect_panel_B1_list = np.vstack(indirect_panel_B1_list)


        if self.save:
            save_path = './results/Big/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.save(save_path + 'big_panel_B1.npy', indirect_panel_B1_list)
            np.save(save_path + 'big_panel_A2.npy', indirect_panel_A2_list)
            print(f'The results have been sucessfully saved in {save_path}')