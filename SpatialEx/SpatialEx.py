"""
SpatialEx training utilities.

This module contains three trainer classes:

- :class:`Train_SpatialEx`: Trains two SpatialEx models (one per slice) and
  evaluates cross-panel prediction quality via cosine similarity, SSIM, PCC,
  and CMD.
- :class:`SpatialExP`: Trains SpatialEx+ with additional regression
  mapping heads in a cycle-style setup to translate between gene panels.
- :class:`SpatialExP_Big`: Trains SpatialEx+ specifically for millons cells.

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


class SpatialEx:
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
                 graph1,
                 graph2, 
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
                 save_path=None
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = graph1
        self.graph2 = graph2
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
        self.save_path = save_path

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars
        
        self.module_HA = Model(self.num_layers, self.in_dim1, self.hidden_dim, self.out_dim1, self.loss_fn, self.device)
        self.module_HB = Model(self.num_layers, self.in_dim2, self.hidden_dim, self.out_dim2, self.loss_fn, self.device)
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

        # H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')      # 注释掉
        self.slice1_dataloader = pp.Build_dataloader(adata1, graph=graph1, graph_norm='hpnn', feat_norm=False,
                                                     prune=[prune, prune], drop_last=False)
        # H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')      # 注释掉
        self.slice2_dataloader = pp.Build_dataloader(adata2, graph=graph2, graph_norm='hpnn', feat_norm=False,
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
    def inference(self, he, graph, panel='panelA'):
        """Predict gene expression for a given panel on a single slice.

        This is a lightweight inference helper that runs the corresponding
        trained model (:attr:`module_HA` for ``panelA`` or :attr:`module_HB` for
        ``panelB``) on the provided histology embedding and spatial graph.

        Parameters
        ----------
        he : array-like
            Histology-derived embedding of shape ``(n_cells, n_he_features)``.
        graph : scipy.sparse.spmatrix or compatible
            Sparse adjacency / hypergraph matrix for the slice.
        panel : {"panelA", "panelB"}, default "panelA"
            Which trained model to use.

        Returns
        -------
        numpy.ndarray
            Predicted expression matrix of shape ``(n_cells, n_genes_in_panel)``.

        Notes
        -----
        If :attr:`save_path` is set, predictions are saved as ``.npy`` files.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        
        if panel == 'panelA':
            self.module_HA.eval()
            panel = self.module_HA.predict(he, graph).detach().cpu().numpy()
        if panel == 'panelB':
            self.module_HB.eval()
            panel = self.module_HB.predict(he, graph).detach().cpu().numpy()

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + panel + '.npy', panel)
            print(f'The results have been sucessfully saved in {self.save_path}')      # 改成保存路径
        return panel

    def auto_inference(self):
        """Run cross-panel prediction for both slices using internal dataloaders.

        The method uses the trained models to predict the *missing* panel for
        each slice:

        - Slice 1: predict panel B using :attr:`module_HB`
        - Slice 2: predict panel A using :attr:`module_HA`

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(panel_1b, panel_2a)`` where:

            - ``panel_1b``: panel-B prediction for slice 1
            - ``panel_2a``: panel-A prediction for slice 2

        Notes
        -----
        If :attr:`save_path` is set, predictions are saved as ``B1.npy`` and
        ``A2.npy``.
        """
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
        panel_1b = panel_1b.values

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
        panel_2a = panel_2a.values

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + 'B1.npy', panel_1b)
            np.save(self.save_path + 'A2.npy', panel_2a)
            print(f'The results have been sucessfully saved in {self.save_path}')      # 改成保存路径
        return panel_1b, panel_2a


class SpatialExP:
    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 platform = 'Xenium',
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
                 save_path=None
                 ):
        """Initialize the SpatialEx+ trainer with cycle-style regression heads.

        This trainer fits two SpatialEx+ backbones (one per slice) and two
        regression mappers (:attr:`rm_AB`, :attr:`rm_BA`) to translate between
        gene panels. During training it optimizes reconstruction losses and
        cycle-consistency-style mapping losses.

        Parameters
        ----------
        adata1, adata2 : AnnData
            Two slices with expression matrices in ``.X`` and histology
            embeddings in ``.obsm['he']``.
        graph1, graph2 : scipy.sparse.spmatrix or compatible
            Spatial graphs for the two slices.
        platform : str, default "Xenium"
            Platform name forwarded to :class:`~model.Model_Plus`.
        seed : int, default 0
            Random seed.
        device : torch.device, optional
            Device on which models and tensors are placed.
        weight_decay : float, default 0
            Weight decay for the optimizer.
        optimizer : str, default "adam"
            Optimizer key understood by :func:`utils.create_optimizer`.
        batch_size : int, default 4096
            Kept for compatibility with other trainers (not used directly here).
        encoder : str, default "hgnn"
            Encoder key (kept for logging/compatibility).
        hidden_dim : int, default 512
            Hidden dimension of the backbone.
        num_layers : int, default 2
            Number of backbone layers.
        epochs : int, default 1000
            Training epochs.
        lr : float, default 0.001
            Learning rate.
        loss_fn : str, default "mse"
            Loss function key.
        num_neighbors : int, default 7
            K for KNN graph construction (kept for compatibility).
        graph_kind : str, default "spatial"
            Graph kind label (kept for compatibility).
        save_path : str or None, optional
            If provided, directory to save inference outputs.
        """
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = pp.sparse_mx_to_torch_sparse_tensor(graph1).to(device)
        self.graph2 = pp.sparse_mx_to_torch_sparse_tensor(graph2).to(device)
        # self.graph1 = graph1
        # self.graph2 = graph2
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
        self.save_path = save_path

        # 空间参数
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind


        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']).to(self.device), torch.Tensor(adata2.obsm['he']).to(self.device)
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X).to(self.device), torch.Tensor(adata2.X).to(self.device)

        self.in_dim1 = adata1.obsm['he'].shape[1]
        self.in_dim2 = adata2.obsm['he'].shape[1]
        self.out_dim1 = adata1.n_vars
        self.out_dim2 = adata2.n_vars

        self.module_HA = Model_Plus(in_dim=self.in_dim1, hidden_dim=self.hidden_dim, out_dim=self.out_dim1, num_layers=self.num_layers,
                                   platform=platform).to(self.device)
        self.module_HB = Model_Plus(in_dim=self.in_dim2, hidden_dim=self.hidden_dim, out_dim=self.out_dim2, num_layers=self.num_layers,
                                   platform=platform).to(self.device)

        self.rm_AB = Regression(self.out_dim1, self.out_dim2, self.out_dim2).to(self.device)
        self.rm_BA = Regression(self.out_dim2, self.out_dim1, self.out_dim1).to(self.device)
        self.models = [self.module_HA, self.module_HB, self.rm_AB, self.rm_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)
    
    def train(self):
        """Train SpatialEx+ backbones and regression mappers.

        The optimization combines:

        1) Per-slice reconstruction losses from :class:`~model.Model_Plus`.
        2) Cross-panel mapping losses for ``A->B`` and ``B->A`` via the regression
           heads.
        3) Cycle-style consistency losses mapping real panel expressions through
           the opposite backbone.

        Returns
        -------
        None
        """
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        self.rm_AB.train()
        self.rm_BA.train()
        print('\n')
        print('=================================== Start training =========================================')
        for epoch in tqdm(range(self.epochs)):
            loss1, _ = self.module_HA(self.HE1, self.graph1, self.panelA1)
            loss2, _ = self.module_HB(self.HE2, self.graph2, self.panelB2)

            panelA2 = self.module_HA.predict(self.HE2, self.graph2, grad=False)
            panelB1 = self.module_HB.predict(self.HE1, self.graph1, grad=False)
            loss3, _ = self.rm_AB(panelA2, self.panelB2)
            loss4, _ = self.rm_BA(panelB1, self.panelA1)

            loss5, _ = self.rm_AB(self.panelA1, panelB1)
            loss6, _ = self.rm_BA(self.panelB2, panelA2)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def inference_direct(self, he, graph, panel='panelA'):
        """Directly predict the specified panel with its corresponding backbone.

        Parameters
        ----------
        he : array-like
            Histology embedding for the query slice.
        graph : scipy.sparse.spmatrix or compatible
            Sparse graph for the query slice.
        panel : {"panelA", "panelB"}, default "panelA"
            Which panel to predict directly.

        Returns
        -------
        numpy.ndarray
            Direct panel prediction of shape ``(n_cells, n_genes_in_panel)``.

        Notes
        -----
        If :attr:`save_path` is set, outputs are saved as
        ``<panel>_direct.npy``.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        
        if panel == 'panelA':
            self.module_HA.eval()
            omics_direct = self.module_HA.predict(he, graph, grad=False)
        if panel == 'panelB':
            self.module_HB.eval()
            omics_direct = self.module_HB.predict(he, graph, grad=False)
            
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + panel + '_direct.npy', omics_direct.detach().cpu().numpy())
            print(f'The results have been sucessfully saved in {self.save_path}')
         
        return omics_direct.detach().cpu().numpy()
        
        
    def inference_indirect(self, he, graph, panel='panelA'):
        """Indirectly infer the missing panel using a regression mapper.

        For ``panelB`` inference, the method first predicts panel A with
        :attr:`module_HA`, then maps to panel B using :attr:`rm_AB`. For
        ``panelA`` inference it uses :attr:`module_HB` followed by
        :attr:`rm_BA`.

        Parameters
        ----------
        he : array-like
            Histology embedding for the query slice.
        graph : scipy.sparse.spmatrix or compatible
            Sparse graph for the query slice.
        panel : {"panelA", "panelB"}, default "panelA"
            Which panel to infer indirectly.

        Returns
        -------
        numpy.ndarray
            Indirect panel prediction of shape ``(n_cells, n_genes_in_target_panel)``.

        Notes
        -----
        If :attr:`save_path` is set, outputs are saved as ``omics.npy``.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        
        if panel == 'panelB':
            self.module_HA.eval()
            self.rm_AB.eval()
            panelA1_direct = self.module_HA.predict(he, graph, grad=False)
            omics_indirect = self.rm_AB.predict(panelA1_direct)
            omics_indirect = omics_indirect.detach().cpu().numpy()
        if panel == 'panelA':
            self.module_HB.eval()
            self.rm_BA.eval()
            panelB2_direct = self.module_HB.predict(he, graph, grad=False)
            omics_indirect = self.rm_BA.predict(panelB2_direct)
            omics_indirect = omics_indirect.detach().cpu().numpy()
            
        if self.save_path:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + 'omics.npy', omics_indirect)
            print(f'The results have been sucessfully saved in {self.save_path}')
        
        return omics_indirect
    
        '''========================= 测试 ========================'''
    def auto_inference(self):        
        """Run indirect cross-panel prediction for both slices.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(panelB1_indirect, panelA2_indirect)`` predictions for slice 1 and
            slice 2 respectively.

        Notes
        -----
        If :attr:`save_path` is set, outputs are saved as ``B1.npy`` and
        ``A2.npy``.
        """
        self.module_HA.eval()
        self.module_HB.eval()
        self.rm_AB.eval()
        self.rm_BA.eval()

        '''PanelB1'''
        panelA1_direct = self.module_HA.predict(self.HE1, self.graph1, grad=False)
        panelB1_indirect = self.rm_AB.predict(panelA1_direct).detach().cpu().numpy()

        '''PanelA2'''
        panelB2_direct = self.module_HB.predict(self.HE2, self.graph2, grad=False)
        panelA2_indirect = self.rm_BA.predict(panelB2_direct).detach().cpu().numpy()

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + 'B1.npy', panelB1_indirect)
            np.save(self.save_path + 'A2.npy', panelA2_indirect)
            print(f'The results have been sucessfully saved in {self.save_path}')

        return panelB1_indirect, panelA2_indirect


class SpatialExP_Big:
    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 num_layers=2,
                 hidden_dim=512,
                 epochs=200,
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
                 save_path=None
                 ):
        """Initialize the large-scale SpatialEx+ trainer using pseudo-spots.

        This variant aggregates single-cell expression into pseudo-spots to
        reduce memory and enable training on very large datasets. It trains a
        shared big backbone (:attr:`model_big`) plus two regression mappers
        (:attr:`model_AB`, :attr:`model_BA`) for cross-panel translation.

        Parameters
        ----------
        adata1, adata2 : AnnData
            Two slices with histology embeddings in ``.obsm['he']`` and
            expression in ``.X``.
        graph1, graph2 : scipy.sparse.spmatrix or compatible
            Spatial graphs for the two slices.
        num_layers : int, default 2
            Number of backbone layers.
        hidden_dim : int, default 512
            Hidden dimension of the backbone.
        epochs : int, default 200
            Number of training epochs.
        seed : int, default 0
            Random seed.
        device : torch.device, optional
            Device to run on.
        weight_decay : float, default 0
            Weight decay for the optimizer.
        optimizer : str, default "adam"
            Optimizer key.
        batch_size : int, default 4096
            Kept for compatibility (batching here is driven by ``batch_num``).
        batch_num : int, default 10
            Number of pseudo-spot batches per epoch.
        encoder : str, default "hgnn"
            Encoder key (kept for compatibility).
        lr : float, default 0.001
            Learning rate.
        loss_fn : str, default "mse"
            Loss function key.
        num_neighbors : int, default 7
            K for KNN (kept for compatibility).
        graph_kind : str, default "spatial"
            Graph kind label (kept for compatibility).
        save_path : str or None, optional
            Directory to save inference outputs.
        """
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = graph1,
        self.graph2 = graph2,
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
        self.save_path = save_path

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        # H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata1 = Generate_pseudo_spot(adata1, all_in=True)
        spot_id = adata1.obs['spot'].values
        head = spot_id[~pd.isna(adata1.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata1.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx1 = sp.coo_matrix((values, (head, tail)), shape=(head.max() + 1, adata1.n_obs)).tocsr()
        self.spot_A1 = torch.Tensor(self.agg_mtx1 @ adata1.X)

        # H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata2 = Generate_pseudo_spot(adata2, all_in=True)
        spot_id = adata2.obs['spot'].values
        head = spot_id[~pd.isna(adata2.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata2.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx2 = sp.coo_matrix((values, (head, tail)), shape=(head.max()+1, adata2.n_obs)).tocsr()
        self.spot_B2 = torch.Tensor(self.agg_mtx2 @ adata2.X)

        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']), torch.Tensor(adata2.obsm['he'])
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X), torch.Tensor(adata2.X)

        self.model_big = Model_Big([graph1, graph2], [self.in_dim1, self.in_dim2], [self.out_dim1, self.out_dim2], num_layers=self.num_layers,
                                   hidden_dim=self.hidden_dim, device=self.device).to(self.device)
        self.model_AB = Regression(self.out_dim1, int(self.out_dim1/2), self.out_dim2).to(self.device)
        self.model_BA = Regression(self.out_dim2, int(self.out_dim1/2), self.out_dim1).to(self.device)
        self.models = [self.model_big, self.model_AB, self.model_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

    def train(self):
        """Train the big model with pseudo-spot batching.

        Each epoch shuffles pseudo-spot indices and iterates over ``batch_num``
        batches. For each batch it:

        1) Computes losses for the shared backbone on both slices.
        2) Generates exchanged predictions and trains regression mappers.
        3) Applies reconstruction-style mapping losses using the aggregation
           matrices.

        Returns
        -------
        None
        """
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
                batch_iter.set_description(f"#Epoch {epoch}, loss: {round(loss.item(), 2)}")

    def inference_indirect(self, he, graph, panel='panelA'):
        """Indirectly infer the missing panel on a query slice in batches.

        Parameters
        ----------
        he : array-like
            Histology embedding for the query slice.
        graph : scipy.sparse.spmatrix or compatible
            Graph for the query slice.
        panel : {"panelA", "panelB"}, default "panelA"
            Target panel to infer.

        Returns
        -------
        numpy.ndarray
            Indirect prediction matrix for the requested panel.

        Notes
        -----
        The computation is split into ``batch_num`` chunks to reduce peak memory.
        If :attr:`save_path` is set, outputs are saved as ``<panel>.npy``.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)

        if panel == 'panelB':
            self.model_big.eval()
            self.model_AB.eval()

            obs_index1 = list(range(he.shape[0]))
            obs_index2 = list(range(self.HE2.shape[0]))
            batch_size1 = int(np.ceil(he.shape[0]/self.batch_num))
            batch_size2 = int(np.ceil(self.HE2.shape[0]/self.batch_num))
            batch_iter = tqdm(range(self.batch_num), leave=False)
            
            indirect_panel_list = []
            tgt_id1_list = []
            for batch_idx in batch_iter:
                tgt_id1 = obs_index1[batch_idx*batch_size1:min((batch_idx+1)*batch_size1, he.shape[0])]
                tgt_id2 = obs_index2[batch_idx*batch_size2:min((batch_idx+1)*batch_size2, self.HE2.shape[0])]
                
                x_prime = self.model_big.predict([tgt_id1, tgt_id2], [he, self.HE2], exchange=False, which='both')
                panel_A1_predict = x_prime[0]
                indirect_panel_B1 = self.model_AB.predict(panel_A1_predict)
                
                tgt_id1_list = tgt_id1_list + tgt_id1
                indirect_panel_list.append(indirect_panel_B1.detach().cpu().numpy())
            indirect_panel_list = np.vstack(indirect_panel_list)

        if panel == 'panelA':
            self.model_big.eval()
            self.model_BA.eval()

            obs_index1 = list(range(self.HE1.shape[0]))
            obs_index2 = list(range(he.shape[0]))
            batch_size1 = int(np.ceil(self.HE1.shape[0]/self.batch_num))
            batch_size2 = int(np.ceil(he.shape[0]/self.batch_num))
            batch_iter = tqdm(range(self.batch_num), leave=False)
            
            indirect_panel_list = []
            tgt_id2_list = []
            for batch_idx in batch_iter:
                tgt_id1 = obs_index1[batch_idx*batch_size1:min((batch_idx+1)*batch_size1, self.HE1.shape[0])]
                tgt_id2 = obs_index2[batch_idx*batch_size2:min((batch_idx+1)*batch_size2, he.shape[0])]
                
                x_prime = self.model_big.predict([tgt_id1, tgt_id2], [self.HE1, he], exchange=False, which='both')
                panel_B2_predict = x_prime[1]
                indirect_panel_A2 = self.model_BA.predict(panel_B2_predict)
                
                tgt_id2_list = tgt_id2_list + tgt_id2
                indirect_panel_list.append(indirect_panel_A2.detach().cpu().numpy())
            indirect_panel_list = np.vstack(indirect_panel_list)
            
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + panel + '.npy', indirect_panel_list)
            print(f'The results have been sucessfully saved in {self.save_path}')

        return indirect_panel_list


    def auto_inference(self):
        """Run indirect cross-panel prediction for both original slices.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(indirect_panel_B1, indirect_panel_A2)`` predictions.

        Notes
        -----
        If :attr:`save_path` is set, outputs are saved as ``B1.npy`` and
        ``A2.npy``.
        """
        self.model_big.eval()
        self.model_AB.eval()
        self.model_BA.eval()

        obs_index1 = list(range(self.HE1.shape[0]))
        obs_index2 = list(range(self.HE2.shape[0]))
        batch_size1 = int(np.ceil(self.HE1.shape[0]/self.batch_num))
        batch_size2 = int(np.ceil(self.HE2.shape[0]/self.batch_num))
        batch_iter = tqdm(range(self.batch_num), leave=False)

        indirect_panel_B1_list = []
        indirect_panel_A2_list = []
        tgt_id1_list = []
        tgt_id2_list = []
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

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + 'B1.npy', indirect_panel_B1_list)
            np.save(self.save_path + 'A2.npy', indirect_panel_A2_list)
            print(f'The results have been sucessfully saved in {self.save_path}')

        return indirect_panel_B1_list, indirect_panel_A2_list
