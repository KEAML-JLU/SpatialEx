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
from train import Model, Regression
from utils import Compute_metrics, create_optimizer

os.environ['R_HOME'] = '/usr/lib/R'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
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
    parser.add_argument("--sample_name", type=str, default="Human_Breast_Cancer_Rep1")  
    parser.add_argument("--file_path", type=str, default='/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big1/cell_feature_matrix.h5')
    parser.add_argument("--obs_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big1/cells.csv") 
    parser.add_argument("--img_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big1/img_sub2.npy") 
    parser.add_argument("--transform_mtx_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv") 
    parser.add_argument("--scale", type=float, default=0.273812612008366)
    parser.add_argument("--cell_diameter", type=float, default=-1, help = "By physical size (um)")
    parser.add_argument("--resolution", type=float, default=64, help = "By pixels")

    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/Xenium_modality_impute/output/")
    parser.add_argument("--impute_sample_name", type=str, default="Human_Breast_Cancer_Rep2")
    parser.add_argument("--impute_file_path", type=str, default='/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big2/cell_feature_matrix.h5')
    parser.add_argument("--impute_obs_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big2/cells.csv") 
    parser.add_argument("--impute_img_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big2/img_sub2.npy") 
    parser.add_argument("--impute_transform_mtx_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv") 

    parser.add_argument("--num_features", type=int, default=3000)
    parser.add_argument("--seq_tech", type=str, default="Xenium")
    parser.add_argument("--cluster_label", type=str, default= "") 
    parser.add_argument("--num_classes", type=int, default=8, help = "The number of clusters")
    parser.add_argument("--nei_radius", type=int, default=7)
    parser.add_argument("--panel", type=str, default='name')
    parser.add_argument("--zinb", type=float, default=0.25)
    
    # read parameters
    args = parser.parse_args()
    return args

def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(args.device)
    pp.set_random_seed(args.seed)

    '''========================= 训练 ========================'''

    adata = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub2.h5ad')
    selection = pd.read_csv(f'/home/wcy/code/pyFile/Xenium_modality_impute/inputs/panel/Selection_by_{args.panel}.csv', index_col=0)
    selection = selection.loc[adata.var_names]
    genes_1 = selection.index[selection['slice1']].tolist()                                        # 选150个训练    

    """第一个切片"""
    #adata = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub2.h5ad')
    adata = pp.Preprocess_adata(adata, selected_genes=genes_1)                                     # 预处理基因表达数据
    he_representations = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Big1_sub2_uni.npy')
    
    # print(adata.X.shape)
    # print(he_representations.shape)
    # assert 0==1

    # img = np.load("/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big1/img_sub2.npy")
    # he_patches, adata = pp.Tiling_HE_patches(args, adata, img)                                     # 提取单细胞对应的HE图像表征
    # he_representations = pp.Extract_HE_patches_representaion(args, he_patches, store_key='he', adata=adata)
    # np.save("/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Big1_sub2_uni.npy", he_representations)
    
    if args.graph_kind == 'spatial':
        H = pp.Build_hypergraph_spatial_and_HE(args, adata, graph_kind='spatial', normalize=True)    # 构建超图
    elif args.graph_kind == 'he':
        H = pp.Build_hypergraph_spatial_and_HE(args, adata, graph_kind='he', normalize=True)
    elif args.graph_kind == 'all':
        H = pp.Build_hypergraph_spatial_and_HE(args, adata, graph_kind='all', normalize=True)

    exp_mtx = torch.Tensor(adata.X).to(args.device)

    
    """第二片切片"""

    # selection = pd.read_csv(f'/home/wcy/code/pyFile/Xenium_modality_impute/inputs/panel/Selection_by_{args.panel}.csv', index_col=0)        
    genes_2 = selection.index[selection['slice2']].tolist()

    adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub3.h5ad')
    #adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big2_sub2.h5ad')
    #adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big2_sub3.h5ad')
    # img = np.load("/home/wcy/code/datasets/Xenium/Human_Breast_IDC_Big2/img_sub2.npy")
    adata_slice2 = pp.Preprocess_adata(adata_slice2, selected_genes=genes_2)
    he_representations_test = np.load('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Big1_sub3_uni.npy')
    #print(adata_slice2.shape, he_representations_test.shape)

    #adata_slice2 = pp.Preprocess_adata(adata_slice2)
    
    # transform_matrix = pd.read_csv(args.impute_transform_mtx_path, header=None).values                   # 基因表达与HE图像配准
    # adata_slice2 = pp.Register_physical_to_pixel(adata_slice2, transform_matrix, args.scale)
    # he_patches, adata_slice2 = pp.Tiling_HE_patches(args, adata_slice2, img)          # 提取单细胞对应的HE图像表征
    # he_representations_test = pp.Extract_HE_patches_representaion(args, he_patches, store_key='he', adata=adata_slice2)
    # np.save("/home/wcy/code/pyFile/Xenium_modality_impute/inputs/he/Big2_sub2_uni.npy", he_representations_test)
    #adata_slice2.write_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Human_Breast_Cancer_Rep2_uni_resolution64_genes2.h5ad')
    
    #adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Human_Breast_Cancer_Rep2_resnet50_resolution64_genes2.h5ad')
    
    #adata_slice2_rep = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Human_Breast_Cancer_Rep2_uni_resolution64_genes2.h5ad')
    #he_representations_test = adata_slice2_rep.obsm['he']

    if args.graph_kind == 'spatial':
        H_test = pp.Build_hypergraph_spatial_and_HE(args, adata_slice2, graph_kind='spatial', normalize=True) # 构建超图
    elif args.graph_kind == 'he':
        H_test = pp.Build_hypergraph_spatial_and_HE(args, adata_slice2, graph_kind='he', normalize=True)
    elif args.graph_kind == 'all':
        H_test = pp.Build_hypergraph_spatial_and_HE(args, adata_slice2, graph_kind='all', normalize=True)

    exp_mtx_2 = torch.Tensor(adata_slice2.X).to(args.device)

    in_dim = exp_mtx.shape[1]
    in_dim_2 = exp_mtx_2.shape[1]

    module_1 = Model(args, adata, he_representations, H)
    rm_1 = Regression(in_dim, in_dim_2, in_dim_2).to(args.device)

    module_2 = Model(args, adata_slice2, he_representations_test, H_test)    
    rm_2 = Regression(in_dim_2, in_dim, in_dim).to(args.device)

    models = [module_1, module_2, rm_1, rm_2]
    optimizer = create_optimizer(args.optimizer, models, args.lr, args.weight_decay)               # 创建优化器
    

    epoch_iter = tqdm(range(args.epoch))

    module_1.train()
    module_2.train()
    rm_1.train()
    rm_2.train()
    print('\n')
    print('=================================== Start training =========================================')
    for epoch in epoch_iter:

        loss1, panel_1a = module_1()
        loss2, panel_2b = module_2()

        panel_2a = module_1.Predict(args, he_representations_test, H_test)
        panel_1b = module_2.Predict(args, he_representations, H)

        # 利用ground truth进行训练
        loss3 = rm_1(exp_mtx, panel_1b)
        loss4 = rm_1(panel_2a, exp_mtx_2)

        loss5 = rm_2(exp_mtx_2, panel_2a)
        loss6 = rm_2(panel_1b, exp_mtx)

        # 利用推理结果进行训练
        # loss3 = rm_1(panel_1a, panel_1b)
        # loss4 = rm_1(panel_2a, panel_2b)

        # loss5 = rm_2(panel_2b, panel_2a)
        # loss6 = rm_2(panel_1b, panel_1a)

        loss = loss1 + loss2 + 1 * (loss3 + loss4 + loss5 + loss6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.2f}")
        
    '''========================= 测试 ========================'''

    '''在Slice1上预测PanelB'''
    # 直接预测
    module_2.eval()
    panelB1 = module_2.Predict(args, he_representations, H).detach().cpu().numpy()

    # 间接预测
    # module_1.eval()
    # rm_1.eval()
    # panelA1 = module_1.Predict(args, adata.obsm['he'], H)
    # panelB1 = rm_1.Predict(panelA1).detach().cpu().numpy()

    #adata_raw = pp.Read_Xenium(args.file_path, args.obs_path)
    #selection = pd.read_csv(f'/home/wcy/code/pyFile/Xenium_modality_impute/inputs/panel/Selection_by_{args.panel}.csv', index_col=0)        
    #genes_2 = selection.index[selection['slice2']].tolist()

    # img = np.load(args.img_path)
    adata = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub2.h5ad')
    adata = pp.Preprocess_adata(adata, cell_mRNA_cutoff=10, selected_genes=genes_1)            # 不筛除细胞， 构建slice1上的panelB的ground truth                                   
    obs_names = adata.obs_names
    adata = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub2.h5ad')
    adata = pp.Preprocess_adata(adata, cell_mRNA_cutoff=0, selected_genes=genes_2)
    adata = adata[obs_names]


    graph = pp.Build_graph(adata.obsm['spatial'], graph_type='radius', radius=8, apply_normalize='gaussian', type='coo')
    cs_sg, cs_reduce_sg = Compute_metrics(adata.X, panelB1, metric='cosine_similarity', dim=0)        # 分别以单细胞、基因计算余弦相似度和均方根误差
    ssim, ssim_reduce = Compute_metrics(adata.X, panelB1, metric='ssim', dim=0, graph=graph)
    pcc, pcc_reduce = Compute_metrics(adata.X, panelB1, metric='pcc', dim=0)               
    print('Evaluation of the predicted PanelB on Slice1, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce)     
 

    '''在Slice2上预测PanelA'''
    # 直接预测
    module_1.eval()
    panelA2 = module_1.Predict(args, he_representations_test, H_test).detach().cpu().numpy()

    # 间接预测
    # module_2.eval()
    # rm_2.eval()
    # panelB2 = module_2.Predict(args, adata_slice2.obsm['he'], H_test)
    # panelA2 = rm_2.Predict(panelB2).detach().cpu().numpy()

    
    #adata_raw = pp.Read_Xenium(args.impute_file_path, args.impute_obs_path)
    #selection = pd.read_csv(f'/home/wcy/code/pyFile/Xenium_modality_impute/inputs/panel/Selection_by_{args.panel}.csv', index_col=0)        
    #genes_1 = selection.index[selection['slice1']].tolist()

    adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub3.h5ad')
    adata_slice2 = pp.Preprocess_adata(adata_slice2, cell_mRNA_cutoff=10, selected_genes=genes_2)            # 不筛除细胞， 构建slice1上的panelB的ground truth                                   
    obs_names = adata_slice2.obs_names
    adata_slice2 = sc.read_h5ad('/home/wcy/code/pyFile/Xenium_modality_impute/inputs/adata/Big1_sub3.h5ad')
    adata_slice2 = pp.Preprocess_adata(adata_slice2, cell_mRNA_cutoff=0, selected_genes=genes_1)
    adata_slice2 = adata_slice2[obs_names]                                                            

    graph = pp.Build_graph(adata_slice2.obsm['spatial'], graph_type='radius', radius=8, apply_normalize='gaussian', type='coo')
    cs_sg, cs_reduce_sg = Compute_metrics(adata_slice2.X, panelA2, metric='cosine_similarity', dim=0)        # 分别以单细胞、基因计算余弦相似度和均方根误差
    ssim, ssim_reduce = Compute_metrics(adata_slice2.X, panelA2, metric='ssim', dim=0, graph=graph)
    pcc, pcc_reduce = Compute_metrics(adata_slice2.X, panelA2, metric='pcc', dim=0)               
    print('Evaluation of the predicted PanelA on Slice2, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce)   

    # if args.save:
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     np.save(args.output_folder + 'adata/Big1_sub2_PanelB' + current_time + '.npy', panelB1)
    #     np.save(args.output_folder + 'adata/Big1_sub3_PanelA' + current_time + '.npy', panelA2)
    
        
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
