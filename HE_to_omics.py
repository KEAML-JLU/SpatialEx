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
from utils import Compute_metrics, create_optimizer, Generate_pseudo_spot

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
    parser.add_argument("--file_path", type=str, default='/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/cell_feature_matrix.h5')
    parser.add_argument("--obs_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/cells.csv") 
    parser.add_argument("--img_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif") 
    parser.add_argument("--transform_mtx_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv") 
    parser.add_argument("--scale", type=float, default=0.363788)
    parser.add_argument("--cell_diameter", type=float, default=-1, help = "By physical size (um)")
    parser.add_argument("--resolution", type=float, default=64, help = "By pixels")

    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--output_folder", type=str, default="/home/wcy/code/pyFile/Xenium_modality_impute/version_2/output/adata/313_gene/")
    parser.add_argument("--impute_sample_name", type=str, default="Human_Breast_Cancer_Rep2")
    parser.add_argument("--impute_file_path", type=str, default='/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/cell_feature_matrix.h5')
    parser.add_argument("--impute_obs_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/cells.csv") 
    parser.add_argument("--impute_img_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif") 
    parser.add_argument("--impute_transform_mtx_path", type=str, default="/home/wcy/code/datasets/Xenium/Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv") 

    parser.add_argument("--num_features", type=int, default=3000)
    parser.add_argument("--scale_exp", type=bool, default=False)
    parser.add_argument("--prune", type=int, default=10000)
    parser.add_argument("--seq_tech", type=str, default="Xenium")
    parser.add_argument("--cluster_label", type=str, default= "") 
    parser.add_argument("--num_classes", type=int, default=8, help = "The number of clusters")
    parser.add_argument("--nei_radius", type=int, default=7)
    

    # read parameters
    args = parser.parse_args()
    return args

def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    pp.set_random_seed(args.seed)
    
    """第一个切片"""
    # adata1 = pp.Read_Xenium(args.file_path, args.obs_path)    
    # adata1 = pp.Preprocess_adata(adata1, scale=args.scale_exp)                                     # preprocess the gene data
    # img, scale = pp.Read_HE_image(args.img_path)
    # if scale != -1:
    #     args.scale = scale

    # transform_mtx = pd.read_csv(args.transform_mtx_path, header=None).values
    # adata1 = pp.Register_physical_to_pixel(adata1, transform_mtx, scale=args.scale)
    # he_patches, adata1 = pp.Tiling_HE_patches(args, adata1, img)
    # pp.Extract_HE_patches_representaion(args, he_patches, store_key='he', adata=adata1)
    adata1 = sc.read_h5ad('./dataset/Human_Rep1_uni_resolution64_full.h5ad')
    H1 = pp.Build_hypergraph_spatial_and_HE(args, adata1, graph_kind='spatial', normalize=False, type='crs')    # 构建超图
    slice1_dataloader = pp.Build_dataloader(adata1, graph=H1, graph_norm='hpnn', feat_norm=False, 
                                        prune=[args.prune, args.prune], drop_last=False)

    """第二片切片"""
    # adata2 = pp.Read_Xenium(args.impute_file_path, args.impute_obs_path)    
    # adata2 = pp.Preprocess_adata(adata2, scale=args.scale_exp)                                     # 预处理基因表达数据
    # img, scale = pp.Read_HE_image(args.impute_img_path)
    # if scale != -1:
    #     args.scale = scale
    # transform_mtx = pd.read_csv(args.impute_transform_mtx_path, header=None).values
    # adata2 = pp.Register_physical_to_pixel(adata2, transform_mtx, scale=args.scale)
    # he_patches, adata2 = pp.Tiling_HE_patches(args, adata2, img)
    # pp.Extract_HE_patches_representaion(args, he_patches, store_key='he', adata=adata2)

    adata2 = sc.read_h5ad('./dataset/Human_Rep2_uni_resolution64_full.h5ad')
    H2 = pp.Build_hypergraph_spatial_and_HE(args, adata2, graph_kind='spatial', normalize=False, type='crs') # 构建超图
    slice2_dataloader = pp.Build_dataloader(adata2, graph=H2, graph_norm='hpnn', feat_norm=False, 
                                            prune=[args.prune, args.prune], drop_last=False)

    in_dim1 = adata1.obsm['he'].shape[1]
    in_dim2 = adata2.obsm['he'].shape[1]
    out_dim1 = adata1.n_vars
    out_dim2 = adata2.n_vars

    module_HA = Model(args, in_dim=in_dim1, out_dim=out_dim1)
    module_HB = Model(args, in_dim=in_dim2, out_dim=out_dim2) 
    models = [module_HA, module_HB]
    optimizer = create_optimizer(args.optimizer, models, args.lr, args.weight_decay)               # 创建优化器

    module_HA.train()
    module_HB.train()
    print('\n')
    print('=================================== Start training =========================================')
    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
        batch_iter = zip(slice1_dataloader, slice2_dataloader)
        for data1, data2 in batch_iter:
            graph1, he1, panel_1a, selection1 = data1[0]['graph'].to(args.device), data1[0]['he'].to(args.device), data1[0]['exp'].to(args.device), data1[0]['selection']
            graph2, he2, panel_2b, selection2 = data2[0]['graph'].to(args.device), data2[0]['he'].to(args.device), data2[0]['exp'].to(args.device), data2[0]['selection']
            agg_mtx1, agg_exp1 = data1[0]['agg_mtx'].to(args.device), data1[0]['agg_exp'].to(args.device)
            agg_mtx2, agg_exp2 = data2[0]['agg_mtx'].to(args.device), data2[0]['agg_exp'].to(args.device)

            loss1, _ = module_HA(graph1, he1, agg_exp1, agg_mtx1, selection1)
            loss2, _ = module_HB(graph2, he2, agg_exp2, agg_mtx2, selection2)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_iter.set_description(f"#Epoch: {epoch}: train_loss: {loss.item():.2f}")
            
    '''========================= 测试 ========================'''
    module_HA.eval()
    module_HB.eval()
    '''PanelA1'''
    panel_1b = []
    obs_list = []
    for data in slice1_dataloader:
        graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
        panelB1 = module_HB.predict(he, graph).detach().cpu().numpy()
        panel_1b.append(panelB1)
        obs_list = obs_list + obs
    panel_1b = np.vstack(panel_1b)
    panel_1b = pd.DataFrame(panel_1b)
    panel_1b.columns = adata1.var_names
    panel_1b['obs_name'] = obs_list
    panel_1b = panel_1b.groupby('obs_name').mean()

    adata_raw = pp.Read_Xenium(args.file_path, args.obs_path)                                              
    adata_raw = pp.Preprocess_adata(adata_raw, cell_mRNA_cutoff=0, scale=args.scale_exp)            # 不筛除细胞， 构建slice1上的panelB的ground truth                                     
    adata = adata_raw[panel_1b.index]                                        
    #graph = pp.Build_graph(adata.obsm['spatial'], graph_type='radius', radius=8, apply_normalize='gaussian', type='coo')
    graph = pp.Build_graph(adata.obsm['spatial'], graph_type='knn', weighted='gaussian', apply_normalize='row',
                           type='coo')
    
    # gene-level
    cs_sg, cs_reduce_sg = Compute_metrics(adata.X, panel_1b.values, metric='cosine_similarity')        # 分别以单细胞、基因计算余弦相似度和均方根误差
    ssim, ssim_reduce = Compute_metrics(adata.X, panel_1b.values, metric='ssim', graph=graph)
    pcc, pcc_reduce = Compute_metrics(adata.X, panel_1b.values, metric='pcc')
    cmd, cmd_reduce = Compute_metrics(adata.X, panel_1b.values, metric='cmd')
    print('Evaluation of the Slice1 in gene-level, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)     
 
    
    '''Panel2B'''
    panel_2a = []
    obs_list = []
    for data in slice2_dataloader:
        graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
        panel2A = module_HA.predict(he, graph).detach().cpu().numpy()
        panel_2a.append(panel2A)
        obs_list = obs_list + obs
    panel_2a = np.vstack(panel_2a)
    panel_2a = pd.DataFrame(panel_2a)
    panel_2a.columns = adata2.var_names
    panel_2a['obs_name'] = obs_list
    panel_2a = panel_2a.groupby('obs_name').mean()

    adata_raw = pp.Read_Xenium(args.impute_file_path, args.impute_obs_path)                                              
    adata_raw = pp.Preprocess_adata(adata_raw, cell_mRNA_cutoff=0, scale=args.scale_exp)                   # 不筛除细胞，构建slice2上的panelA的ground truth
    adata_slice2 = adata_raw[panel_2a.index]                                                         
    graph = pp.Build_graph(adata_slice2.obsm['spatial'], graph_type='knn', weighted='gaussian', apply_normalize='row', type='coo')
    
    # gene-level
    cs_sg, cs_reduce_sg = Compute_metrics(adata_slice2.X, panel_2a.values, metric='cosine_similarity')        # 分别以单细胞、基因计算余弦相似度和均方根误差
    ssim, ssim_reduce = Compute_metrics(adata_slice2.X, panel_2a.values, metric='ssim', graph=graph)
    pcc, pcc_reduce = Compute_metrics(adata_slice2.X, panel_2a.values, metric='pcc')               
    print('Evaluation of the Slice2 in gene-level, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce)
    


    # if args.save:
    #     panel_1b.to_csv(args.output_folder + 'ours_313_panelB1_no_norm.csv')
    #     panel_2a.to_csv(args.output_folder + 'ours_313_panelA2_no_norm.csv')

    # '''PanelB1'''
    # panel_1b = []
    # obs_list = []
    # for data in slice1_dataloader:
    #     graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
    #     panelB1 = module_HB.predict(he, graph).detach().cpu().numpy()
    #     panel_1b.append(panelB1)
    #     obs_list = obs_list + obs
    # panel_1b = np.vstack(panel_1b)
    # panel_1b = pd.DataFrame(panel_1b)
    # panel_1b.columns = adata2.var_names
    # panel_1b['obs_name'] = obs_list
    # panel_1b = panel_1b.groupby('obs_name').mean()

    # adata_raw = pp.Read_Xenium(args.file_path, args.obs_path)                                              
    # adata_raw = pp.Preprocess_adata(adata_raw, cell_mRNA_cutoff=0, scale=args.scale_exp)            # 不筛除细胞， 构建slice1上的panelB的ground truth                                     
    # adata = adata_raw[panel_1b.index]                                        
    # graph = pp.Build_graph(adata.obsm['spatial'], graph_type='radius', radius=8, apply_normalize='gaussian', type='coo')
    # cs_sg, cs_reduce_sg = Compute_metrics(adata.X, panel_1b.values, metric='cosine_similarity', dim=0)        # 分别以单细胞、基因计算余弦相似度和均方根误差
    # ssim, ssim_reduce = Compute_metrics(adata.X, panel_1b.values, metric='ssim', dim=0, graph=graph)
    # pcc, pcc_reduce = Compute_metrics(adata.X, panel_1b.values, metric='pcc', dim=0)
    # cmd, cmd_reduce = Compute_metrics(adata.X, panel_1b.values, metric='cmd', dim=0)
    # print('Evaluation of the predicted PanelB on Slice1, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce) 

    
    # '''PanelA2'''
    # panel_2a = []
    # obs_list = []
    # for data in slice2_dataloader:
    #     graph, he, obs = data[0]['graph'].to(args.device), data[0]['he'].to(args.device), data[0]['obs']
    #     panel2A = module_HA.predict(he, graph).detach().cpu().numpy()
    #     panel_2a.append(panel2A)
    #     obs_list = obs_list + obs
    # panel_2a = np.vstack(panel_2a)
    # panel_2a = pd.DataFrame(panel_2a)
    # panel_2a.columns = adata1.var_names
    # panel_2a['obs_name'] = obs_list
    # panel_2a = panel_2a.groupby('obs_name').mean()

    # adata_raw = pp.Read_Xenium(args.impute_file_path, args.impute_obs_path)                                              
    # adata_raw = pp.Preprocess_adata(adata_raw, cell_mRNA_cutoff=0, scale=args.scale_exp)                   # 不筛除细胞，构建slice2上的panelA的ground truth
    # adata_slice2 = adata_raw[panel_2a.index]                                                         
    # graph = pp.Build_graph(adata_slice2.obsm['spatial'], graph_type='radius', radius=8, apply_normalize='gaussian', type='coo')
    # cs_sg, cs_reduce_sg = Compute_metrics(adata_slice2.X, panel_2a.values, metric='cosine_similarity', dim=0)        # 分别以单细胞、基因计算余弦相似度和均方根误差
    # ssim, ssim_reduce = Compute_metrics(adata_slice2.X, panel_2a.values, metric='ssim', dim=0, graph=graph)
    # pcc, pcc_reduce = Compute_metrics(adata_slice2.X, panel_2a.values, metric='pcc', dim=0)
    # cmd, cmd_reduce = Compute_metrics(adata_slice2.X, panel_2a.values, metric='cmd', dim=0)
    # print('Evaluation of the predicted PanelB on Slice1, cosine similarity: ', cs_reduce_sg, ' ssim: ', ssim_reduce, ' pcc: ', pcc_reduce, ' cmd: ', cmd_reduce) 

    
    # if args.save:
    #     panel_1b.to_csv(args.output_folder + 'vanilla_' + args.sample_name + '.csv')
    #     panel_2a.to_csv(args.output_folder + 'vanilla_' + args.impute_sample_name + '.csv')





if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
