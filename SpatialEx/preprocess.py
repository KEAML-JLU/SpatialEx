import torch
import random
import itertools
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
import scipy.sparse as sp
from cellpose import models
import torch.nn.functional as F
from PIL import Image, ImageFile
import xml.etree.ElementTree as ET
from .utils import create_ImageEncoder
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader
from .utils import Generate_pseudo_spot
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from sklearn.preprocessing import normalize, StandardScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Read_Xenium(h5_path, obs_path):
    adata = sc.read_10x_h5(h5_path)
    adata.obs = pd.read_csv(obs_path, index_col=0)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def Preprocess_adata(adata, cell_mRNA_cutoff=10, selected_genes=None, scale=False):
    adata.var_names_make_unique()
    if selected_genes is not None:
        adata = adata[:, selected_genes]
    sc.pp.filter_cells(adata, min_counts=cell_mRNA_cutoff)
    adata.layers['raw'] = adata.X.copy()

    if isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.todense().A
    if scale:
        gene_min = adata.X.min(0)
        gene_max = adata.X.max(0)
        adata.var['min'] = gene_min
        adata.var['max'] = gene_max
        adata.X = adata.X - gene_min
        adata.X = adata.X / (gene_max - gene_min + 1e-12)
    else:
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
    return adata


def Read_HE_image(img_path, suffix='.ome.tif'):
    scale = -1
    if suffix == '.ome.tif':
        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        metadata = ome_tif.ome_metadata
        ome_tif.close()

        root = ET.fromstring(metadata)
        namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        pixels_element = root.find('.//ome:Pixels', namespace)  # 找到 <Pixels> 标签
        if pixels_element is not None:
            pixels_attributes = pixels_element.attrib
            for attr, value in pixels_attributes.items():
                if attr == 'PhysicalSizeX' or attr == 'PhysicalSizeX':
                    scale = float(value)
                    break
    elif suffix == '.png' or suffix == '.jpg':
        image = Image.open(img_path)
        image_data = np.array(image)
    elif suffix == '.tif':
        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        ome_tif.close()
    else:
        print("Only support '.ome.tif', '.png' or '.jpg' file currently.")
    return image_data, scale


def Crop_HE_image(img, crop):
    img = img[crop[0]:crop[1], crop[2]:crop[3]]
    return img


def Cell_segmentation(img, percent_threshold=0.2, pixel_threshold=190, device='cpu',
                      patch_size=512, scale=1, min_size=50, chan=[1, 0], diameter=12, flow_threshold=1.0):
    '''
    Utilize Cellpose to perform cell segmentation on histological image, important params will be surrounded by stars
    **percent_threshold, pixel_threshold**:  int,  The patch in which 1-percent_threshold percent of the pixels
                                                   take values greater than 190 will be regared as bounaries
    patch_size:     int,    Tiling the big img into (patch_size, patch_size, 3) dims to reduce memory consumption
    min_size:       int,    Any ROI covering an area smaller than this value will be discarded
    chan:           list,   Cell is distinguishable in chan[0]+1 channel of the img, and chan[1] is the backfround channel, for nuclei model, chan[1]=0
    **diameter**:   int,    The median of the diameter of detected cells
    flow_threshold: float,  Any cell with a probability smaller than this value whill be discarded
    scale:          float,  Reduce the big img to reduce memory consumption
    '''

    print('======================== Estimating cells from histological image ===========================')
    image_tensor = torch.Tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)  # 转化成(C, W, H), 添加批次维度
    _, _, height, width = image_tensor.shape  # 添加padding，确保图像边缘不被丢弃
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    image_tensor_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=255)

    patches = image_tensor_padded.unfold(2, patch_size, patch_size).unfold(3, patch_size,
                                                                           patch_size)  # 使用unfold将图像切分成patches
    patches = patches.squeeze().permute(1, 2, 3, 4, 0).numpy().astype(int)

    model = models.Cellpose(gpu=True, model_type='nuclei', device=device)
    total_num = 0.0
    cell_list = []
    for i in tqdm(range(patches.shape[0])):
        row_list = []
        for j in range(patches.shape[1]):
            img = patches[i, j]
            if np.percentile(img, percent_threshold) > pixel_threshold:
                nuclei_masks = np.zeros_like(img[..., 0])
            else:
                nuclei_masks, _, _, _ = model.eval(255 - img, flow_threshold=flow_threshold, min_size=min_size,
                                                   diameter=diameter, channels=chan, invert=False)
                cell_num = nuclei_masks.max()  # 记录当前细胞数量
                nuclei_masks = nuclei_masks.astype(np.float64)  # 细胞数量很大会溢出
                nuclei_masks[nuclei_masks != 0] += total_num  # 提取的细胞序号加上之前的细胞总数, 得到不重复的细胞序号
                total_num = total_num + cell_num  # 更新细胞总数
            row_list.append(nuclei_masks)

        row_list = np.hstack(row_list)
        cell_list.append(row_list)
    cell_list = np.vstack(cell_list)  # 组装回一张图
    print(total_num, ' cells detected')

    coords = np.array(np.nonzero(cell_list)).T
    values = cell_list[tuple(coords.T)]
    df = pd.DataFrame(coords, columns=['row', 'col'])
    df['value'] = values
    mean_coor = df.groupby('value').mean()  # 计算每个细胞的位置
    return np.round(mean_coor.values)


def Estimate_scale(physical_coor, pixel_coor, unit=100, reduce='median'):
    '''
    Assuming a linear correspondence between the physical location and the histological image,
    estimate how much physical distance by um corresponds to one pixel.
    unit:   int,    unit of the measure by um
    '''
    spatial = pd.DataFrame(np.hstack([physical_coor, pixel_coor]))
    spatial.columns = ['x', 'y', 'px', 'py']
    spatial_diff = spatial[1:].values - spatial[:-1].values
    spatial_diff = pd.DataFrame(spatial_diff)
    spatial_diff.columns = spatial.columns

    scale_x = spatial_diff['x'] * unit / (spatial_diff['px'] + 1e-6)
    scale_y = spatial_diff['y'] * unit / (spatial_diff['py'] + 1e-6)
    scale_x = scale_x[scale_x != 0.0]
    scale_y = scale_y[scale_y != 0.0]
    if reduce == 'median':
        scale = [np.median(scale_x.values), np.median(scale_y.values)]
    elif reduce == 'mean':
        scale = [np.mean(scale_x.values), np.mean(scale_y.values)]
    return scale


def Register_physical_to_pixel(adata, transform_matrix, scale=1,
                               raw_key=['x_centroid', 'y_centroid'],
                               matrix_type='pixel2phsical',  # 'pixel2phsical'或者'physical2pixel'
                               prefix='image'):
    scale_old = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
    scale = scale / scale_old
    transform_matrix = transform_matrix * scale
    transform_matrix[-1, -1] = 1

    if matrix_type == 'pixel2phsical':
        transform_matrix = np.linalg.inv(transform_matrix)

    x = adata.obs[raw_key[0]].values
    y = adata.obs[raw_key[1]].values
    ones = np.ones_like(x)
    coor_raw = np.vstack([x, y, ones])

    coor_new = (transform_matrix @ coor_raw)[:2, :]
    image_coor = np.round(coor_new).astype(int)
    adata.obsm[prefix + '_coor'] = image_coor.T
    adata.obs[prefix + '_col'] = image_coor[0]
    adata.obs[prefix + '_row'] = image_coor[1]
    return adata


def Tiling_HE_patches(resolution, adata, img, key='image_coor'):  # iStar中说，单细胞大小约为8um*8um
    print('======================== Tiling HE patches for each single cells ===========================')
    patch_radius = int(resolution / 2.0)
    print("patch radius is ", patch_radius)

    outlier_cells = np.unique(np.where(adata.obsm[key] < patch_radius)[0])
    if len(outlier_cells) != 0:
        print('Remove the outlier cells, and Anndata file was reduced!')
        inlier_cells = set(np.arange(adata.n_obs)) - set(outlier_cells)
        adata = adata[list(inlier_cells)]
    he_patches = [0] * adata.n_obs
    adata.obsm[key] = adata.obsm[key].astype(int)
    for i in tqdm(range(adata.n_obs)):
        x, y = adata.obsm[key][i]
        he_patches[i] = torch.tensor(img[y - patch_radius: y + patch_radius, x - patch_radius:x + patch_radius])
    return torch.stack(he_patches, dim=0) / 255.0, adata


def Tiling_HE_patches_by_coor(resolution, coor, img, col_name=['col', 'row']):  # iStar中说，单细胞大小约为8um*8um
    print('======================== Tiling HE patches for each single cells ===========================')

    patch_radius = int(resolution / 2.0)
    outlier_cells1 = np.where(coor < patch_radius)[0]
    outlier_cells2 = np.where(coor[col_name[0]] > (img.shape[1] - patch_radius))[0]
    outlier_cells3 = np.where(coor[col_name[1]] > (img.shape[0] - patch_radius))[0]
    outlier_cells = np.unique(np.hstack([outlier_cells1, outlier_cells2, outlier_cells3]))
    if len(outlier_cells) != 0:
        print('Remove the outlier cells, and Anndata file was reduced!')
        inlier_cells = set(np.arange(coor.shape[0])) - set(outlier_cells)
        coor = coor.iloc[list(inlier_cells)]
    he_patches = [0] * coor.shape[0]
    for i in tqdm(range(coor.shape[0])):
        x, y = coor.iloc[i][col_name[0]], coor.iloc[i][col_name[1]]
        he_patches[i] = torch.tensor(img[y - patch_radius: y + patch_radius, x - patch_radius:x + patch_radius])
    return torch.stack(he_patches, dim=0) / 255.0, coor


def Extract_HE_patches_representaion(he_patches, store_key=None, adata=None, skip_embedding=False, img_batch_size=64, image_encoder='uni', device='cuda'):
    if he_patches.dim() == 3:
        he_patches = he_patches.unsqueeze(0)  # 如果不是batch，则补充batch维度
    if he_patches.size(1) != 3:
        he_patches = he_patches.permute(0, 3, 1, 2)  # 通道维度放前面, (batch, channel, x, y)

    print('====================== Extracting HE representations for each cell =========================')
    if image_encoder == 'uni':
        print("The image encoder is uni")
        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )])
    elif image_encoder == 'conch':
        print("The image encoder is conch")
        preprocess = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ])
    elif image_encoder == 'gigapath':
        print("The image encoder is gigapath")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif image_encoder == 'phikon':
        print("The image encoder is phikon")
        preprocess = AutoImageProcessor.from_pretrained("owkin/phikon", do_rescale=False)
    elif image_encoder in ['resnet50', 'resnet101', 'resnet152']:
        print(f"The image encoder is {image_encoder}")
        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )])
    else:
        print("The image encoder is not implemented")
        raise NotImplementedError

    representaions = []
    batch_num = int(np.ceil(he_patches.size(0) / img_batch_size))
    if not skip_embedding:
        model = create_ImageEncoder(image_encoder)
        model.to(device)
        model.eval()

        for i in tqdm(range(batch_num)):
            if image_encoder == 'phikon':
                img_tensor = preprocess(
                    he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))],
                    return_tensors="pt")
            else:
                img_tensor = preprocess(
                    he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))].to(device))
            with torch.no_grad():
                if image_encoder == 'conch':
                    features = model.encode_image(img_tensor, proj_contrast=False,
                                                  normalize=False).squeeze().detach().cpu().numpy()
                elif image_encoder == 'phikon':
                    img_tensor = img_tensor['pixel_values'].to(device)
                    features = model(img_tensor).last_hidden_state[:, 0, :].detach().cpu().numpy()
                else:
                    features = model(img_tensor).squeeze().detach().cpu().numpy()
                representaions.append(features)
    else:
        for i in tqdm(range(batch_num)):
            img_tensor = preprocess(
                he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))])
            representaions.append(img_tensor)

    representaions = np.vstack(representaions)
    if isinstance(store_key, str):
        adata.obsm[store_key] = representaions
    return adata


def Build_graph(x, weighted=False, symmetric=False, graph_type='radius', metric='euclidean', self_loop=True,
                radius=50, num_neighbors=50, apply_normalize='none', sigma=0.01, type='coo'):
    '''
    graph_type: str,    'radius' will connect the nodes within the radius(50 by default), 
                        'knn' will connect the num_neighbors(50 by default) nearest neighbors
    
    weighted:   str,    'reciprocal' will lead to calculate the reciprocal of the distance as a weight
                        'gaussian' will lead to calculate the gaussian kernel as a weight, sigma is 1.5 by default
                        'none' will generate a binary adj

    symmetric:  bool    False will directly return the adj
                        True will makes adj[i, j] = adj[j, i]
    '''
    metric = metric.lower()
    apply_normalize = apply_normalize.lower()
    graph_type = graph_type.lower()

    if metric == 'cosine':
        x = normalize(x, norm='l2')  # L2归一化后的欧氏距离相当于余弦距离

    tree = BallTree(x)  # 仅支持欧氏距离
    if graph_type == 'radius':
        tail_list, distances = tree.query_radius(x, r=radius, return_distance=True)
    elif graph_type == 'knn':
        distances, tail_list = tree.query(x, k=num_neighbors)

    head_list = []
    head_list = [head_list + [i] * len(tail_list[i]) for i in range(len(tail_list))]
    head_list = list(itertools.chain.from_iterable(head_list))
    tail_list = list(itertools.chain.from_iterable(tail_list))

    if not weighted:
        distances = np.ones_like(head_list)
    elif isinstance(weighted, str):
        distances = np.array(list(itertools.chain.from_iterable(distances)))
        if metric == 'cosine':
            distances = (distances * distances) / 2  # 余弦相似度是利用欧氏距离间接计算的，1-cos(x,y) = d^2/2
        if weighted == 'reciprocal':
            distances = 1 / distances
        elif weighted == 'gaussian':
            distances = np.exp(-(distances ** 2) / 2 * sigma * sigma) / (2 * np.pi * sigma * sigma)

    adj = sp.coo_matrix((distances, (head_list, tail_list)), shape=(x.shape[0], x.shape[0]))  # 用稀疏矩阵构建，方便后续计算

    if not self_loop:
        adj = adj.tocsr()
        adj.setdiag(0)

    if symmetric:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if apply_normalize == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif apply_normalize == 'gcn':
        D = np.squeeze(adj.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ adj @ D

    if type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif type == 'csr':
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()
    return adj


def Build_graph_for_high_dim_feat(x, weighted=False, num_neighbors=50, device='cpu', batch_size=4096,
                                  apply_normalize=False, type='coo'):
    '''Use GPU to compute cosine similarity and build knn graph'''

    print('======================== Build adj on high dimensional features =========================')
    x = torch.Tensor(x).to(device)
    x = F.normalize(x, p=2, dim=1)

    tail_list = []
    distance_list = []
    batch_num = int(np.ceil(x.shape[0] / batch_size))
    for i in tqdm(range(batch_num)):
        cosine_sim = x[i * batch_size:min((i + 1) * batch_size, x.shape[0])] @ x.T
        topk_values, topk_indices = torch.topk(cosine_sim, k=num_neighbors)
        tail_list.append(topk_indices)
        distance_list.append(topk_values)

    tail_list = torch.vstack(tail_list).reshape(-1).detach().cpu().numpy()
    head_list = np.arange(x.shape[0]).repeat(num_neighbors)
    if weighted:
        distance = torch.vstack(distance_list).reshape(-1)
        distance = (1 / distance).detach().cpu().numpy()
    else:
        distance = np.ones_like(head_list)

    adj = sp.coo_matrix((distance, (head_list, tail_list)), shape=(x.shape[0], x.shape[0]))

    if apply_normalize == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif apply_normalize == 'gcn':
        D = np.squeeze(adj.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ adj @ D

    if type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif type == 'crs':
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()
    return adj


def Build_hypergraph(x, metric='euclidean', graph_type='knn', radius=50, num_neighbors=50,
                     self_loop=True, normalize=False, edge_weight=None, type='coo'):
    H = Build_graph(x, metric=metric, graph_type=graph_type, radius=radius, num_neighbors=num_neighbors,
                    self_loop=self_loop, type=type)
    H = H.T
    if normalize:
        H = normalize_graph(H, edge_weight, type='hpnn')
    return H


def Build_hypergraph_spatial_and_HE(adata, num_neighbors=7, batch_size=4096, normalize=False, graph_kind='spatial',
                                    type='coo', device="cpu"):
    if graph_kind.lower() == 'spatial':
        H1 = Build_graph(adata.obsm['spatial'], graph_type='knn', num_neighbors=num_neighbors, type=type)
        H = H1.T
    elif graph_kind.lower() == 'he':
        H2 = Build_graph_for_high_dim_feat(adata.obsm['he'], num_neighbors=num_neighbors,
                                           device=device, batch_size=batch_size)
        H = H2.T
    elif graph_kind.lower() == 'all':
        H1 = Build_graph(adata.obsm['spatial'], graph_type='knn', num_neighbors=num_neighbors)
        H2 = Build_graph_for_high_dim_feat(adata.obsm['he'], num_neighbors=num_neighbors,
                                           device=device, batch_size=batch_size)
        H = sp.hstack([H1.T, H2.T])
    else:
        assert False

    if normalize:
        H = normalize_graph(H, type='hpnn')

    if type == 'coo':
        if not isinstance(H, sp.coo_matrix):
            H = H.tocoo()
    elif type == 'crs':
        if not isinstance(H, sp.csr_matrix):
            H = H.tocsr()
    return H

# 超图归一化
def normalize_hypergraph(H, edge_weight=None):
    DE = np.squeeze(H.sum(0).A)
    DV = np.squeeze(H.sum(1).A)
    DE = sp.diags(np.power(DE.astype(float), -1), offsets=0, format='csr')
    DV = sp.diags(np.power(DV.astype(float), -0.5), offsets=0, format='csr')
    if edge_weight != None:
        W = sp.diags(np.squeeze(edge_weight), offsets=0, format='csr')
    else:
        W = sp.diags(np.ones(shape=(H.shape[1])), offsets=0, format='csr')
    return DV @ H @ W @ DE @ H.T @ DV


# 常规图归一化
def normalize_graph(H, edge_weight=None, type='gcn'):
    if type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors)
    elif type == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        adj = adj.multiply(normalization_factors)
    elif type == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))  # 列归一化
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))  # 行归一化
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif type == 'gcn':
        D = np.squeeze(H.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ H @ D
    elif type == 'hpnn':
        DE = np.squeeze(H.sum(0).A)
        DV = np.squeeze(H.sum(1).A)
        DE = sp.diags(np.power(DE.astype(float), -1), offsets=0, format='csr')
        DV = sp.diags(np.power(DV.astype(float), -0.5), offsets=0, format='csr')
        if edge_weight != None:
            W = sp.diags(np.squeeze(edge_weight), offsets=0, format='csr')
        else:
            W = sp.diags(np.ones(shape=(H.shape[1])), offsets=0, format='csr')
        adj = DV @ H @ W @ DE @ H.T @ DV

    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx, return_mtx=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if return_mtx:
        return torch.sparse.FloatTensor(indices, values, shape)
    else:
        return indices, values, shape


def Build_dataloader(adata, graph, batch_size=1, ori=False, graph_norm='hpnn', feat_norm=False,
                     shuffle=True, prune=[10000, 10000], drop_last=False):
    '''
    ori:         bool,           Whether or not a count matrix is required
    
    prune:       list(int, int), How many pixels on the image correspond to the rows and columns of small slices

    batch_size:  int,            When the **prune** is not given, the **prune** would be estimated according to the **batch_size**, 
                                 so that approximately the number of small slices equals to **batch_size**.
    '''
    dataset = Xenium_HBRC_overlap(adata, graph, ori=ori, prune=prune, graph_norm=graph_norm, feat_norm=feat_norm,
                                  drop_last=drop_last)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    return train_loader


def custom_collate_fn(batch):
    return batch


class Xenium_HBRC_overlap(torch.utils.data.Dataset):
    def __init__(self, adata, graph, ori=False, graph_norm='hpnn', feat_norm=False, prune=[3000, 3000],
                 drop_last=False):
        super(Xenium_HBRC_overlap, self).__init__()

        self.ori = ori
        _, _, adata = Generate_pseudo_spot(adata, all_in=True)
        spot_id = adata.obs['spot'].values
        head = spot_id[~pd.isna(adata.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata.obs['spot']))[0]
        values = np.ones_like(tail)
        agg_mtx = sp.coo_matrix((values, (head, tail)), shape=(head.max() + 1, adata.n_obs)).tocsr()

        row = adata.obs['x_centroid'].values  # 从0开始比较好计算
        col = adata.obs['y_centroid'].values
        exp = torch.Tensor(adata.X)
        he = adata.obsm['he']
        if feat_norm:
            scaler = StandardScaler()
            he = scaler.fit_transform(he)
        he = torch.Tensor(he)

        x_cat = row // prune[0]  # 确定方形区域左上角坐标
        y_cat = col // prune[1]
        x_min, x_max = x_cat.min(), x_cat.max()
        y_min, y_max = y_cat.min(), y_cat.max()
        x_cat = x_cat.astype(int).astype(str)
        y_cat = y_cat.astype(int).astype(str)
        idx = np.char.add(np.char.add(x_cat, '*'), y_cat)
        idx_cat = np.unique(idx)
        self.idx_cat = idx_cat.copy()

        if row.max() < prune[0]:
            prune[0] = row.max()
        if col.max() < prune[1]:
            prune[1] = col.max()

        self.roi_dict = {}
        self.selection_dict = {}
        self.exp_dict = {}
        self.he_dict = {}
        self.graph_dict = {}
        self.agg_dict = {}
        self.agg_exp_dict = {}
        self.obs_dict = {}
        for name in idx_cat:
            x = int(name.split('*')[0]) * prune[0]
            y = int(name.split('*')[-1]) * prune[1]

            selection = (row > x + 0.5 * prune[0]) & (row < (x + 1.5 * prune[0])) & (col > y + 0.5 * prune[1]) & (
                    col < (y + 1.5 * prune[1]))
            if not selection.sum():  # 如果没有中心区域则跳过
                self.idx_cat = self.idx_cat[self.idx_cat != name]
                continue

            self.roi_dict[name] = (row > x) & (row < (x + 2 * prune[0])) & (col > y) & (col < (y + 2 * prune[1]))
            self.he_dict[name] = he[self.roi_dict[name]]
            self.obs_dict[name] = adata.obs_names[self.roi_dict[name]].tolist()

            sub_graph = normalize_graph(graph[self.roi_dict[name]][:, self.roi_dict[name]], type=graph_norm)
            self.graph_dict[name] = sparse_mx_to_torch_sparse_tensor(sub_graph)

            '''选择部分计算损失'''
            if int(name.split('*')[0]) > x_min:
                selection_x_left = (row > x + 0.5 * prune[0])
            else:
                selection_x_left = np.ones_like(row).astype(bool)

            if int(name.split('*')[0]) < x_max:
                selection_x_right = (row < (x + 1.5 * prune[0]))
            else:
                selection_x_right = np.ones_like(row).astype(bool)

            if int(name.split('*')[-1]) > y_min:
                selection_y_down = (col > y + 0.5 * prune[1])
            else:
                selection_y_down = np.ones_like(col).astype(bool)

            if int(name.split('*')[-1]) < y_max:
                selection_y_up = (col < (y + 1.5 * prune[1]))
            else:
                selection_y_up = np.ones_like(col).astype(bool)

            selection = selection_x_left & selection_x_right & selection_y_down & selection_y_up
            self.exp_dict[name] = exp[selection]  # 在整个切片上计算损失的部分
            self.agg_dict[name] = sparse_mx_to_torch_sparse_tensor(agg_mtx[:, selection])
            self.agg_exp_dict[name] = torch.sparse.mm(self.agg_dict[name], self.exp_dict[name])  # 在整个切片上计算损失的部分
            self.selection_dict[name] = selection[self.roi_dict[name]]  # 在ROI中计算损失的部分

        self.id2name = dict(enumerate(self.idx_cat))

        if ori:
            ori = adata.layer['ori'].X.todense()
            sfs = ori / np.median(np.squeeze(ori))
            sfs = torch.Tensor(np.squeeze(self.sfs.A))

            self.ori_dict = {}
            self.sfs_dict = {}
            for idx in self.idx_cat:
                selection = self.selection[idx]
                self.ori_dict[idx] = torch.Tensor(ori[selection].todense())
                self.sfs_dict[idx] = sfs[selection]

    def __getitem__(self, index):
        ID = self.id2name[index]
        data = {}
        data['he'] = self.he_dict[ID]
        data['graph'] = self.graph_dict[ID]
        data['exp'] = self.exp_dict[ID]
        data['selection'] = self.selection_dict[ID]
        data['obs'] = self.obs_dict[ID]
        data['agg_mtx'] = self.agg_dict[ID]
        data['agg_exp'] = self.agg_exp_dict[ID]
        if self.ori:
            data['ori'] = self.ori_dict[ID]
            data['sfs'] = self.sfs_dict[ID]
        return data

    def __len__(self):
        return len(self.idx_cat)
