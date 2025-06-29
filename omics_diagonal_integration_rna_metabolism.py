import os
import warnings
import argparse

import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
#from captum.attr import IntegratedGradients

import preprocess as pp
from train import Regression, Model_modified
from utils import create_optimizer, Compute_metrics, Compute_MoransI

os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cpu'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)
        
def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=4)
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
    parser.add_argument("--epoch", type=int, default=600, help="number of training epochs")
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
    parser.add_argument("--num_neighbors", type=int, default=2)
    parser.add_argument("--graph_kind", type=str, default='spatial')
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--cell_diameter", type=float, default=-1, help = "By physical size (um)")
    parser.add_argument("--resolution", type=float, default=580, help = "By pixels")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/multiomics/")

    parser.add_argument("--sample_name1", type=str, default="rna_V11L12-109_C1")  
    parser.add_argument("--file_path1", type=str, default='/home/wcy/code/datasets/Visium/Multi_modality/rna_V11L12-109_C1.h5ad')
    parser.add_argument("--img_path1", type=str, default="/home/wcy/code/datasets/Visium/Multi_modality/220506_MSi_V11L12-109_C1.jpg") 
    parser.add_argument("--sample_name2", type=str, default="metabolite_V11L12-109_B1")
    parser.add_argument("--file_path2", type=str, default='/home/wcy/code/datasets/Visium/Multi_modality/New/metabolite_V11L12-109_B1.h5ad')
    parser.add_argument("--img_path2", type=str, default="/home/wcy/code/datasets/Visium/Multi_modality/220506_MSi_V11L12-109_B1.jpg") 

    parser.add_argument("--num_features", type=int, default=[1000, 50])
    parser.add_argument("--scale_exp", type=bool, default=False)
    parser.add_argument("--platform", type=str, default="Visium")

    parser.add_argument("--loss_num", type=int, default=6)
    parser.add_argument("--use_dgi", type=bool, default=True)

    
    # read parameters
    args = parser.parse_args()
    return args

def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    pp.set_random_seed(args.seed)
    
    """第一个切片"""
    '''1. 基因表达预处理'''
    C1_rna = sc.read_h5ad(args.file_path1)
    C1_rna.var_names_make_unique()
    sc.pp.filter_genes(C1_rna, min_cells=1)
    C1_rna.var["mt"] = C1_rna.var_names.str.startswith("mt-")
    C1_rna = C1_rna[:, ~C1_rna.var["mt"]]
    sc.pp.log1p(C1_rna)
    C1_rna = pp.Preprocess_adata(C1_rna, cell_mRNA_cutoff=0, scale=True)
    if isinstance(C1_rna.X, sp.csr_matrix):
        C1_rna.X = C1_rna.X.A
    adj = pp.Build_graph(C1_rna.obsm['spatial'], graph_type='knn').todense().A
    moransI = Compute_MoransI(C1_rna, adj)
    mz_selected = C1_rna.var_names[np.argsort(moransI)[-args.num_features[0]:]]
    C1_rna = C1_rna[:, mz_selected]
    panelA1 = torch.Tensor(C1_rna.X).to(args.device)
    np.save(args.output_folder + 'rna_C1_gt.npy', C1_rna.X)

    '''2. HE图像切分与表征提取'''
    # image_coor = np.zeros([C1_rna.n_obs, 2])
    # image_coor[:, 0] = C1_rna.obsm['spatial'][:, 1].copy()
    # image_coor[:, 1] = C1_rna.obsm['spatial'][:, 0].copy()
    # C1_rna.obsm['image_coor'] = image_coor
    # img, _ = pp.Read_HE_image(args.img_path1, suffix='.jpg')
    # he_patches, C1_rna = pp.Tiling_HE_patches(args, C1_rna, img, key='spatial')
    # he1 = pp.Extract_HE_patches_representaion(args, he_patches)
    he1 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/C1_uni_res580.npy')
    node_feat1 = torch.Tensor(he1).to(args.device)

    '''3. 空间图构建'''
    # H1 = pp.Build_hypergraph(C1_rna.obsm['spatial'], graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr', normalize=True)
    H1 = pp.Build_hypergraph(C1_rna.obsm['spatial'], graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr')
    H1 = pp.sparse_mx_to_torch_sparse_tensor(H1).to(args.device)

    '''第二个切片'''
    '''1. 基因表达预处理'''
    B1_metabolite = sc.read_h5ad(args.file_path2)
    B1_metabolite.var_names = B1_metabolite.var['metabolism'].values
    B1_metabolite.var_names_make_unique()
    sc.pp.filter_genes(B1_metabolite, min_cells=1)
    B1_metabolite = pp.Preprocess_adata(B1_metabolite, cell_mRNA_cutoff=0, scale=True)
    if isinstance(B1_metabolite.X, sp.csr_matrix):
        B1_metabolite.X = B1_metabolite.X.A
    adj = pp.Build_graph(B1_metabolite.obsm['spatial'], graph_type='knn').todense().A 
    moransI = Compute_MoransI(B1_metabolite, adj)
    mz_selected = B1_metabolite.var_names[np.argsort(moransI)[-args.num_features[1]:]]
    B1_metabolite = B1_metabolite[:, mz_selected]    
    panelB2 = torch.Tensor(B1_metabolite.X).to(args.device)
    np.save(args.output_folder + 'metabolite_B1_gt.npy', B1_metabolite.X)
    

    '''2. HE图像切分与表征提取'''
    # image_coor = np.zeros([B1_rna.n_obs, 2])
    # image_coor[:, 0] = B1_rna.obsm['spatial'][:, 1].copy()
    # image_coor[:, 1] = B1_rna.obsm['spatial'][:, 0].copy()
    # B1_rna.obsm['image_coor'] = image_coor
    # img, _ = pp.Read_HE_image(args.img_path2, suffix='.jpg')
    # he_patches, B1_metabolite = pp.Tiling_HE_patches(args, B1_metabolite, img, key='spatial')
    # he2 = pp.Extract_HE_patches_representaion(args, he_patches)
    he2 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/B1_uni_res580.npy')
    node_feat2 = torch.Tensor(he2).to(args.device)

    '''3. 空间图构建,聚合图构建'''
    # H2 = pp.Build_hypergraph(B1_metabolite.obsm['spatial'], graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr', normalize=True)
    H2 = pp.Build_hypergraph(B1_metabolite.obsm['spatial'], graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr')
    H2 = pp.sparse_mx_to_torch_sparse_tensor(H2).to(args.device)

    in_dim1 = node_feat1.shape[1]
    in_dim2 = node_feat2.shape[1]
    out_dim1 = C1_rna.n_vars
    out_dim2 = B1_metabolite.n_vars

    model_HA = Model_modified(args, in_dim1, args.hidden_dim, out_dim1, platform=args.platform).to(args.device)
    model_HB = Model_modified(args, in_dim2, args.hidden_dim, out_dim2, platform=args.platform).to(args.device)
    models = [model_HA, model_HB]

    if args.loss_num > 2:
        model_AB = Regression(out_dim1, out_dim2, out_dim2, platform=args.platform).to(args.device)
        model_BA = Regression(out_dim2, out_dim1, out_dim1, platform=args.platform).to(args.device)
        models = models + [model_AB, model_BA]

    optimizer = create_optimizer(args.optimizer, models, args.lr, args.weight_decay)

    print('================================ Trian ================================')
    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
            loss1, _ = model_HA(node_feat1, H1, panelA1)
            loss2, _ = model_HB(node_feat2, H2, panelB2)
            loss = loss1 + loss2

            if args.loss_num > 2:
                panelA2 = model_HA.predict(node_feat2, H2, grad=False)
                panelB1 = model_HB.predict(node_feat1, H1, grad=False)
                loss3, _ = model_AB(panelA2, panelB2)
                loss4, _ = model_BA(panelB1, panelA1)
                loss = loss + loss3 + loss4

            if args.loss_num > 4:
                loss5, _ = model_AB(panelA1, panelB1)
                loss6, _ = model_BA(panelB2, panelA2)
                loss = loss + loss5 + loss6

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            # epoch_iter.set_description(f"#Epoch {epoch},loss1:{round(loss1.item(),2)},loss2:{round(loss2.item(),2)},loss3:{round(loss3.item(),2)},loss4:{round(loss4.item(),2)}")

    model_HA.eval()
    model_HB.eval()
    model_AB.eval()
    model_BA.eval()
    
    # multiomics spatial domain analysis
    _, panelA1_pre = model_HA(node_feat1, H1, panelA1)
    _, panelB2_pre = model_HB(node_feat2, H2, panelB2)


    '''测试'''
    panelA1_direct = model_HA.predict(node_feat1, H1, grad=False)
    panelB2_direct = model_HB.predict(node_feat2, H2, grad=False)  
    panelA2_direct = model_HA.predict(node_feat2, H2, grad=False)
    panelB1_direct = model_HB.predict(node_feat1, H1, grad=False)   
    panelB1_indirect = model_AB.predict(panelA1_direct)
    panelA2_indirect = model_BA.predict(panelB2_direct)
    panelA2_direct = panelA2_direct.detach().cpu().numpy()
    panelB1_direct = panelB1_direct.detach().cpu().numpy()
    panelA2_indirect = panelA2_indirect.detach().cpu().numpy()
    panelB1_indirect = panelB1_indirect.detach().cpu().numpy()
    
    '''Compute_metrics'''
    C1_metabolite = sc.read_h5ad('/home/wcy/code/datasets/Visium/Multi_modality/New/metabolite_V11L12-109_C1.h5ad')
    C1_metabolite.var_names = C1_metabolite.var['metabolism'].values
    C1_metabolite.var_names_make_unique()
    C1_metabolite = C1_metabolite[:, B1_metabolite.var_names]
    C1_metabolite = C1_metabolite[C1_rna.obs_names]
    C1_metabolite = pp.Preprocess_adata(C1_metabolite, cell_mRNA_cutoff=0, scale=True)
    if isinstance(C1_metabolite.X, sp.csr_matrix):
        C1_metabolite.X = C1_metabolite.X.A   

    B1_rna = sc.read_h5ad('/home/wcy/code/datasets/Visium/Multi_modality/New/rna_V11L12-109_B1.h5ad')
    B1_rna.var_names_make_unique()
    B1_rna = B1_rna[:, C1_rna.var_names]
    B1_rna = B1_rna[B1_metabolite.obs_names]
    sc.pp.log1p(B1_rna)
    B1_rna = pp.Preprocess_adata(B1_rna, cell_mRNA_cutoff=0, scale=True)
    if isinstance(B1_rna.X, sp.csr_matrix):
        B1_rna.X = B1_rna.X.A  

    graph = pp.Build_graph(C1_metabolite.obsm['spatial'], graph_type='knn', num_neighbors=50, weighted='gaussian', apply_normalize='row', type='coo')
    ssim, ssim_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_direct.copy(), metric='ssim', graph=graph, reduce='mean')
    pcc, pcc_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_direct.copy(), metric='pcc', reduce='mean')
    cmd, cmd_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_direct.copy(), metric='cmd', reduce='mean')
    print('Directed metabolite: ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)

    ssim, ssim_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_indirect.copy(), metric='ssim', graph=graph, reduce='mean')
    pcc, pcc_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_indirect.copy(), metric='pcc', reduce='mean')
    cmd, cmd_reduce = Compute_metrics(C1_metabolite.X.copy(), panelB1_indirect.copy(), metric='cmd', reduce='mean')
    print('Inirected metabolite: ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)

    graph = pp.Build_graph(B1_rna.obsm['spatial'], graph_type='knn', num_neighbors=50, weighted='gaussian', apply_normalize='row', type='coo')
    ssim, ssim_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_direct.copy(), metric='ssim', graph=graph, reduce='mean')
    pcc, pcc_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_direct.copy(), metric='pcc', reduce='mean')
    cmd, cmd_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_direct.copy(), metric='cmd', reduce='mean')
    print('Direct rna: ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)

    ssim, ssim_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_indirect.copy(), metric='ssim', graph=graph, reduce='mean')
    pcc, pcc_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_indirect.copy(), metric='pcc', reduce='mean')
    cmd, cmd_reduce = Compute_metrics(B1_rna.X.copy(), panelA2_indirect.copy(), metric='cmd', reduce='mean')
    print('Indirected rna: ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)

    if args.save:
        np.save(args.output_folder + 'metabolite_C1_direct.npy', panelB1_direct)
        np.save(args.output_folder + 'metabolite_C1_indirect.npy', panelB1_indirect)
        np.save(args.output_folder + 'rna_B1_direct.npy', panelA2_direct) 
        np.save(args.output_folder + 'rna_B1_indirect.npy', panelA2_indirect)
        
        # for multiomics spatial domain analysis
        np.save(args.output_folder + 'metabolite_B1_predict.npy', panelB2_pre.detach().cpu().numpy())
        np.save(args.output_folder + 'rna_C1_predict.npy', panelA1_pre.detach().cpu().numpy())

    '''Out'''
    B1_coor = pd.read_csv('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/V11L12-109_B1_cell_coor_out_spot.csv', index_col=0)
    # img_path = '/home/wcy/code/datasets/Visium/Multi_modality/220506_MSi_V11L12-109_B1.jpg'
    # img, _ = pp.Read_HE_image(img_path, suffix='.jpg')
    # patches, B1_coor = pp.Tiling_HE_patches_by_coor(args, B1_coor, img, ['image_col', 'image_row'])
    # he_B1 = pp.Extract_HE_patches_representaion(args, patches)
    he_B1 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/B1_out_uni_res580.npy')
    he_B1 = torch.Tensor(he_B1).to(args.device)
    # H1 = pp.Build_hypergraph(B1_coor.values, graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr', normalize=True)
    H1 = pp.Build_hypergraph(B1_coor.values, graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr')
    H1 = pp.sparse_mx_to_torch_sparse_tensor(H1).to(args.device)
    
    C1_coor = pd.read_csv('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/V11L12-109_C1_cell_coor_out_spot.csv', index_col=0)
    # img_path = '/home/wcy/code/datasets/Visium/Multi_modality/220506_MSi_V11L12-109_C1.jpg'
    # img, _ = pp.Read_HE_image(img_path, suffix='.jpg')    
    # patches, C1_coor = pp.Tiling_HE_patches_by_coor(args, C1_coor, img, ['image_col', 'image_row'])
    # he_C1 = pp.Extract_HE_patches_representaion(args, patches)
    he_C1 = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/C1_out_uni_res580.npy')
    he_C1 = torch.Tensor(he_C1).to(args.device)
    # H2 = pp.Build_hypergraph(C1_coor.values, graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr', normalize=True)
    H2 = pp.Build_hypergraph(C1_coor.values, graph_type='knn', num_neighbors=args.num_neighbors, self_loop=True, type='csr')
    H2 = pp.sparse_mx_to_torch_sparse_tensor(H2).to(args.device)

    panelA1_direct = model_HA.predict(he_B1, H1, grad=False)
    panelB2_direct = model_HB.predict(he_C1, H2, grad=False)  
    panelA2_direct = model_HA.predict(he_C1, H2, grad=False)
    panelB1_direct = model_HB.predict(he_B1, H1, grad=False)   
    panelA1_indirect = model_BA.predict(panelB1_direct)
    panelB2_indirect = model_AB.predict(panelA2_direct)
    panelB1_indirect = model_AB.predict(panelA1_direct)
    panelA2_indirect = model_BA.predict(panelB2_direct)

    panelA1_direct = panelA1_direct.detach().cpu().numpy()
    panelB2_direct = panelB2_direct.detach().cpu().numpy()
    panelA1_indirect = panelA1_indirect.detach().cpu().numpy()
    panelB2_indirect = panelB2_indirect.detach().cpu().numpy()
    panelA2_direct = panelA2_direct.detach().cpu().numpy()
    panelB1_direct = panelB1_direct.detach().cpu().numpy()
    panelA2_indirect = panelA2_indirect.detach().cpu().numpy()
    panelB1_indirect = panelB1_indirect.detach().cpu().numpy()

    if args.save:
        np.save(args.output_folder + 'metabolite_C1_out_direct.npy', panelB1_direct)
        np.save(args.output_folder + 'metabolite_C1_out_indirect.npy', panelB1_indirect)
        np.save(args.output_folder + 'rna_B1_out_direct.npy', panelA2_direct) 
        np.save(args.output_folder + 'rna_B1_out_indirect.npy', panelA2_indirect)

        np.save(args.output_folder + 'rna_C1_out_direct.npy', panelA1_direct)
        np.save(args.output_folder + 'rna_C1_out_indirect.npy', panelA1_indirect)
        np.save(args.output_folder + 'metabolite_B1_out_direct.npy', panelB2_direct) 
        np.save(args.output_folder + 'metabolite_B1_out_indirect.npy', panelB2_indirect)

if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
