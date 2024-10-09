import numpy as np
import pandas as pd
import os
import scanpy as sc
from tool.transform_tool import t_or_f
import GRAS4T
from data_read_tool import build_her2st_data, build_stereo_seq_data

### param setting ###

# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=str, default='True', help="Whether to use GPU (set 'True' or 'False')")
parser.add_argument('--platform', type=str, default='10X', help="Platform used for the dataset (e.g., '10X', etc.)")
parser.add_argument('--dataset_name', type=str, default='DLPFC', help="Name of the dataset to be used (e.g., 'DLPFC', etc.)")
parser.add_argument('--slice', type=str, default='151507', help="Slice ID from the dataset")
parser.add_argument('--have_label', type=str, default='False', help="Whether the dataset has labels (set 'True' or 'False')")
parser.add_argument('--num_cluster', type=str, default='None', help="Number of clusters for clustering")

# data preprocessing setting
parser.add_argument('--use_image', type=str, default='True', help="Whether to use image data during preprocessing (set 'True' or 'False')")
parser.add_argument('--save_path', type=str, default='./Results', help='Location where tissue images are saved after being tiled')
parser.add_argument('--cnnType_', type=str, default='ResNet50',
                    help="Type of CNN model used for processing images (e.g., 'ResNet50', etc.)")
parser.add_argument('--n_top_genes', type=int, default=3000, help="Number of top genes selected")

# adj construction setting
parser.add_argument('--k_X_', type=int, default=10, help="Number of nearest neighbors considered for the gene expression matrix")
parser.add_argument('--k_C_', type=int, default=10, help="Number of nearest neighbors considered for the spatial location ")
parser.add_argument('--weight_', type=float, default=0.5, help="Weight for adjacency matrix combination")

# data augmentation settings
parser.add_argument('--aug1', type=str, default='mask', help="Type of node augmentation applied (e.g., 'mask')")
parser.add_argument('--aug2', type=str, default='HS_image', help="Type of edge augmentation applied (e.g., 'HS_image', 'edge')")
parser.add_argument('--drop_percent1', type=float, default=0.1, help="Dropout percentage applied in the node augmentation")
parser.add_argument('--drop_percent2', type=float, default=0.1, help="Dropout percentage applied in the edge augmentation")
parser.add_argument('--image_k', type=int, default=100, help="Number of neighbors considered for HS_image augmentations")

# training param setting
parser.add_argument('--nb_epochs_', type=int, default=500, help="number of epochs")
parser.add_argument('--lr_', type=float, default=0.001, help="Learning rate")
parser.add_argument('--patience_', type=int, default=20, help="Patience for early stopping during training")
parser.add_argument('--interval_num', type=int, default=100, help="Number of interval steps to update the self-expression matrix")
parser.add_argument('--is_select_neighbor', type=str, default='False', help="whether to use post-processing tools")
parser.add_argument('--select_neighbor', type=int, default=5, help="number of neighbors selected in post-processing")
parser.add_argument('--weight_lg', type=float, default=1.0, help="Weight for the global loss")
parser.add_argument('--weight_lc', type=float, default=1.0, help="Weight for the subspace loss")
parser.add_argument('--weight_recon', type=float, default=1.0, help="Weight for the reconstruction loss")

# clustering param setting
parser.add_argument('--cluster_method', type=str, default='mclust', help="Clustering method")

args = parser.parse_args()

# another training param setting
seed = 2023
use_gpu = t_or_f(args.use_gpu)

#### read data ####

print("*" * 50)
print("START GRAS4T | use_gpu:{}, seed:{}".format(use_gpu, seed))
print("*" * 50)

# read adata
if args.platform == '10X':
    input_dir = './data/{}/{}/'.format(args.platform, args.dataset_name) + args.slice
    adata = sc.read_visium(path=input_dir, count_file=args.slice + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    # extract feature from H&E image
    use_image = t_or_f(args.use_image)
    if use_image:
        image_feat_path = "./data/{}/{}/image_feature/{}-{}-image_feat.npy".format(args.platform, args.dataset_name,
                                                                                   args.slice, args.cnnType_)
        image_feat = np.load(image_feat_path)
        adata.obsm["image_feat"] = image_feat

elif args.platform == 'MERFISH':
    im_dict = {-0.29: 'f29', -0.24: 'f24', -0.19: 'f19', -0.14: 'f14', -0.09: 'f09', -0.04: 'f04',
               0.01: '01', 0.06: '06', 0.11: '11', 0.16: '16', 0.21: '21', 0.26: '26'}
    file_path = './data/{}/{}/'.format(args.platform, args.dataset_name) + 'slice-{}.h5ad'.format(
        im_dict[float(args.slice)])
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()

elif args.platform == 'Spatial_Transcriptomics':
    input_dir = './data/{}/{}/{}'.format(args.platform, args.dataset_name, args.slice)
    file_name = "{}_{}".format(args.dataset_name, args.slice)
    adata = build_her2st_data(input_dir, file_name)
    adata.var_names_make_unique()

elif args.platform == 'STARmap':
    input_dir = './data/{}/{}/'.format(args.platform, args.dataset_name) + 'slice-{}.h5ad'.format(args.slice)
    adata = sc.read(input_dir)
    adata.var_names_make_unique()

elif args.platform == 'Stereo-seq':
    input_dir = './data/{}/{}/'.format(args.platform, args.dataset_name)
    adata = build_stereo_seq_data(input_dir)
    adata.var_names_make_unique()

print("adata info:\n", adata)
print("--" * 25)

# get annotation
have_label = t_or_f(args.have_label)
if have_label:
    if (args.platform == '10X') & (args.dataset_name == 'DLPFC'):
        Ann_df = pd.read_csv(
            os.path.join('data/{}/{}/{}_annotations/'.format(args.platform, args.dataset_name, args.dataset_name),
                         args.slice + '_truth.txt'),
            sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    elif (args.platform == '10X') & (args.dataset_name == 'V1_Breast_Cancer_Block_A_Section_1'):
        Ann_df = pd.read_csv(os.path.join('data/{}/{}'.format(args.platform, args.dataset_name), 'metadata.tsv'),
                             sep='\t', header=0, index_col='ID')
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'fine_annot_type']
    elif args.platform == 'MERFISH':
        adata.obs['Ground Truth'] = adata.obs['cell_type']
    elif args.platform == 'STARmap':
        adata.obs['Ground Truth'] = adata.obs.label
    else:
        pass

GRAS4T.run(args, seed=seed, use_gpu=use_gpu, adata=adata, have_label=have_label,
           hid_units_=[64],
           regu_coef=1,
           use_decoder_emb=False,
           activation='prelu',
           muti_act_1='origin',
           muti_act_2='prelu',
           default_resolution=1.5,
           )
