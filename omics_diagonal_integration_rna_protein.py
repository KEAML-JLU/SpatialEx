import os
import warnings
import argparse
from datetime import datetime

import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm

import preprocess as pp
from train import Model_modified, Regression
from utils import create_optimizer, sparse_mx_to_torch_sparse_tensor

os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=9)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=0.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--linear_prob", action="store_true", default=True)
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=True)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--img_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4096)

    # adjustable parameters
    parser.add_argument("--image_encoder", type=str, default="uni")
    parser.add_argument("--encoder", type=str, default="hgnn")
    parser.add_argument("--decoder", type=str, default="linear")
    parser.add_argument("--hidden_dim", type=int, default=512, help="number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--epoch", type=int, default=500, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument("--beta_l", type=float, default=1, help="`pow`inddex for `weighted_mse` loss")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--mask_gene_rate", type=float, default=0.8)
    parser.add_argument("--replace_rate", type=float, default=0.05)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--warm_up", type=int, default=50)
    parser.add_argument("--norm", type=str, default="batchnorm")

    # File parameter
    parser.add_argument("--num_neighbors", type=int, default=7)
    parser.add_argument("--graph_kind", type=str, default='spatial')
    parser.add_argument("--sample_name", type=str, default="Human_Breast_Cancer_Rep2_RNA")
    parser.add_argument("--h5_path1", type=str,
                        default='/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/cell_feature_matrix.h5')
    parser.add_argument("--obs_path1", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/cells.csv")
    parser.add_argument("--img_path1", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif")
    parser.add_argument("--trans_mtx_path1", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv")
    parser.add_argument("--scale", type=float, default=0.363788)
    parser.add_argument("--cell_diameter", type=float, default=-1, help="By physical size (um)")
    parser.add_argument("--resolution", type=float, default=64, help="By pixels")

    parser.add_argument("--sample_name2", type=str, default="Human_Breast_Cancer_Rep1_protein")
    parser.add_argument("--h5_path2", type=str,
                        default='/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/cell_protein_matrix.h5ad')
    parser.add_argument("--obs_path2", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/cells.csv")
    parser.add_argument("--img_path2", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif")
    parser.add_argument("--trans_mtx_path2", type=str,
                        default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv")

    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/")
    parser.add_argument("--num_features", type=int, default=3000)
    parser.add_argument("--scale_exp", type=bool, default=False)
    parser.add_argument("--prune", type=int, default=100000)
    parser.add_argument("--seq_tech", type=str, default="Xenium")
    parser.add_argument("--cluster_label", type=str, default="")
    parser.add_argument("--num_classes", type=int, default=8, help="The number of clusters")
    parser.add_argument("--nei_radius", type=int, default=7)

    # read parameters
    args = parser.parse_args()
    return args


def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    pp.set_random_seed(args.seed)

    """第一个切片"""
    adata1 = pp.Read_Xenium(args.h5_path1, args.obs_path1)
    adata1 = pp.Preprocess_adata(adata1)
    img, scale = pp.Read_HE_image(args.img_path1)
    if scale > 0:
        args.scale = scale
    trans_mtx = pd.read_csv(args.trans_mtx_path1, header=None).values
    adata1 = pp.Register_physical_to_pixel(adata1, trans_mtx, args.scale)
    he_patches, adata1 = pp.Tiling_HE_patches(args, adata1, img)
    # rep = pp.Extract_HE_patches_representaion(args, he_patches, 'he', adata1)
    adata1.obsm['he'] = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Rep2_rna.npy')
    H = pp.Build_hypergraph_spatial_and_HE(args, adata1, graph_kind='spatial', normalize=False, type='crs')  # 构建超图
    slice1_dataloader = pp.Build_dataloader(adata1, graph=H, graph_norm='hpnn', feat_norm=False,
                                            prune=[args.prune, args.prune], drop_last=False)

    """第二片切片""" 
    adata2 = sc.read_h5ad(args.h5_path2)
    adata2.var_names = adata2.var_names.astype(str)
    adata2.obs_names = adata2.obs_names.astype(str)
    obs = pd.read_csv(args.obs_path2, index_col=0)
    obs.index = obs.index.astype(str)
    adata2 = adata2[obs.index]
    adata2.obs = obs
    adata2.obsm['spatial'] = adata2.obs[['x_centroid', 'y_centroid']].values
    adata2.var_names_make_unique()
    sc.pp.scale(adata2)

    # img, scale = pp.Read_HE_image(args.img_path2)
    # if scale > 0:
    #     args.scale = scale   
    # trans_mtx = pd.read_csv(args.trans_mtx_path2, header=None).values
    # adata2 = pp.Register_physical_to_pixel(adata2, trans_mtx, args.scale)
    # he_patches, adata2 = pp.Tiling_HE_patches(args, adata2, img)
    # rep = pp.Extract_HE_patches_representaion(args, he_patches, 'he', adata2)   
    adata2.obsm['he'] = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Rep1_protein.npy')
    H2 = pp.Build_hypergraph_spatial_and_HE(args, adata2, graph_kind='spatial', normalize=False, type='crs')  # 构建超图
    slice2_dataloader = pp.Build_dataloader(adata2, graph=H2, graph_norm='hpnn', feat_norm=False,
                                            prune=[args.prune, args.prune], drop_last=False)


    in_dim1 = adata1.obsm['he'].shape[1]
    in_dim2 = adata2.obsm['he'].shape[1]
    out_dim1 = adata1.n_vars
    out_dim2 = adata2.n_vars
    module_HA = Model_modified(args, in_dim=in_dim1, hidden_dim=args.hidden_dim, out_dim=out_dim1).to(args.device)
    module_HB = Model_modified(args, in_dim=in_dim2, hidden_dim=args.hidden_dim, out_dim=out_dim2).to(args.device)
    # rm_AB = Regression(out_dim1, out_dim2, out_dim2).to(args.device)
    # rm_BA = Regression(out_dim2, out_dim2, out_dim2).to(args.device)
    rm_AB = Regression(out_dim1 + in_dim1, out_dim2, out_dim2).to(args.device)
    rm_BA = Regression(out_dim2 + in_dim2, out_dim1, out_dim1).to(args.device)
    models = [module_HA, module_HB, rm_AB, rm_BA]
    optimizer = create_optimizer(args.optimizer, models, args.lr, args.weight_decay)  # 创建优化器

    module_HA.train()
    module_HB.train()
    rm_AB.train()
    rm_BA.train()
    print('\n')
    print('=================================== Start training =========================================')
    for epoch in tqdm(range(args.epoch)):
        batch_iter = zip(slice1_dataloader, slice2_dataloader)
        for data1, data2 in batch_iter:
            graph1, he1, panel_1a, selection1 = data1[0]['graph'].to(args.device), data1[0]['he'].to(args.device), \
                                                data1[0]['exp'].to(args.device), data1[0]['selection']
            graph2, he2, panel_2b, selection2 = data2[0]['graph'].to(args.device), data2[0]['he'].to(args.device), \
                                                data2[0]['exp'].to(args.device), data2[0]['selection']
            agg_mtx1, agg_exp1 = data1[0]['agg_mtx'].to(args.device), data1[0]['agg_exp'].to(args.device)
            agg_mtx2, agg_exp2 = data2[0]['agg_mtx'].to(args.device), data2[0]['agg_exp'].to(args.device)

            loss1, _ = module_HA(he1, graph1, agg_exp1, agg_mtx1)
            loss2, _ = module_HB(he2, graph2, agg_exp2, agg_mtx2)
            panel_2a = module_HA.predict(he2, graph2)
            panel_1b = module_HB.predict(he1, graph1)

            loss3, _ = rm_AB(torch.hstack([panel_1a, he1]), torch.spmm(agg_mtx1, panel_1b[selection1]), agg_mtx1)
            loss4, _ = rm_AB(torch.hstack([panel_2a, he2]), agg_exp2, agg_mtx2)

            loss5, _ = rm_BA(torch.hstack([panel_2b, he2]), torch.spmm(agg_mtx2, panel_2a[selection2]), agg_mtx2)
            loss6, _ = rm_BA(torch.hstack([panel_1b, he1]), agg_exp1, agg_mtx1)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    '''========================= 测试 ========================'''
    module_HA.eval()
    module_HB.eval()
    rm_AB.eval()
    rm_BA.eval()

    '''indirect PanelB1'''
    panelB1_list = []
    obs_list = []
    for data in slice1_dataloader:
        graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
        panelA1 = module_HA.predict(he, graph)
        panelB1 = rm_AB.predict(torch.hstack([panelA1, he])).detach().cpu().numpy()
        panelB1_list.append(panelB1)
        obs_list = obs_list + obs
    panelB1 = np.vstack(panelB1_list)
    panelB1 = pd.DataFrame(panelB1)
    panelB1.columns = adata2.var_names
    panelB1['obs_name'] = obs_list
    panelB1 = panelB1.groupby('obs_name').mean() 

    '''indirect PanelA2'''
    panelA2_list = []
    obs_list = []
    for data in slice2_dataloader:
        graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
        panelB2 = module_HB.predict(he, graph)
        panelA2 = rm_BA.predict(torch.hstack([panelB2, he])).detach().cpu().numpy()
        panelA2_list.append(panelA2)
        obs_list = obs_list + obs
    panelA2 = np.vstack(panelA2_list)
    panelA2 = pd.DataFrame(panelA2)
    panelA2.columns = adata1.var_names
    panelA2['obs_name'] = obs_list
    panelA2 = panelA2.groupby('obs_name').mean() 

    if args.save:
        panelA2.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/inner_indirect_A2_residual.csv')
        panelB1.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/inner_indirect_B1_residual.csv')

    '''Out'''
    he1 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/HBRC_Rep1_Out_uni.npy')
    he2 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/HBRC_Rep2_Out_uni.npy')
    he1 = torch.Tensor(he1).to(args.device)
    he2 = torch.Tensor(he2).to(args.device)

    obs1 = pd.read_csv('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/HBRC_Rep1_cell_coor.csv', index_col=0)
    obs2 = pd.read_csv('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/HBRC_Rep2_cell_coor.csv', index_col=0)
    H1 = pp.Build_hypergraph(obs1.values, graph_type='knn', num_neighbors=args.num_neighbors, normalize=True)
    H2 = pp.Build_hypergraph(obs2.values, graph_type='knn', num_neighbors=args.num_neighbors, normalize=True)
    graph1 = sparse_mx_to_torch_sparse_tensor(H1, device=args.device)
    graph2 = sparse_mx_to_torch_sparse_tensor(H2, device=args.device)

    '''indirect PanelB1'''
    panelA1 = module_HA.predict(he1, graph1)
    panelB1 = rm_AB.predict(torch.hstack([panelA1, he1])).detach().cpu().numpy()
    panelB1 = pd.DataFrame(panelB1)
    panelB1.columns = adata2.var_names

    '''indirect PanelA2'''
    panelB2 = module_HB.predict(he2, graph2)
    panelA2 = rm_BA.predict(torch.hstack([panelB2, he2])).detach().cpu().numpy()
    panelA2 = pd.DataFrame(panelA2)
    panelA2.columns = adata1.var_names 

    if args.save:
        panelA2.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/outer_indirect_A2_residual.csv')
        panelB1.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/outer_indirect_B1_residual.csv')

    '''indirect PanelB2'''
    panelA2 = module_HA.predict(he2, graph2)
    panelB2 = rm_AB.predict(torch.hstack([panelA2, he2])).detach().cpu().numpy()
    panelB2 = pd.DataFrame(panelB2)
    panelB2.columns = adata2.var_names

    '''indirect PanelA1'''
    panelB1 = module_HB.predict(he1, graph1)
    panelA1 = rm_BA.predict(torch.hstack([panelB1, he1])).detach().cpu().numpy()
    panelA1 = pd.DataFrame(panelA1)
    panelA1.columns = adata1.var_names 

    if args.save:
        panelA1.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/outer_indirect_A1_residual.csv')
        panelB2.to_csv('/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/Rep1_multimodality/outer_indirect_B2_residual.csv')


if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
