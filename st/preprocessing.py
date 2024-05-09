# -*- coding: utf-8 -*-
# __author__ = "GoatBishop"
# Email:guiyang0928@126.com

import scanpy as sc
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy import stats
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import random
from sklearn.decomposition import PCA

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms


class image_feature:
    def __init__(
            self,
            adata,
            pca_components=50,
            cnnType='ResNet50',
            verbose=False,
            seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(
            self,
    ):

        if self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Resnet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model

    def extract_image_feat(
            self,
    ):

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                          transforms.RandomInvert(),
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                          transforms.RandomSolarize(random.uniform(0, 1)),
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2),
                                                  shear=(-0.3, 0.3, -0.3, 0.3)),
                          transforms.RandomErasing()
                          ]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(mean=[0.54, 0.51, 0.68], 
        #                   std =[0.25, 0.21, 0.16])]
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
        # model.fc = torch.nn.LeakyReLU(0.1)
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")

        with tqdm(total=len(self.adata),
                  desc="Extract image feature",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]", ) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype="int32")
                spot_slice = spot_slice.astype(np.float32)
                tensor = img_to_tensor(spot_slice)
                tensor = tensor.resize_(1, 3, 224, 224)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata


def get_X(adata,
          n_top_genes=5000,
          is_pac=False,
          pca_n_comps=200):
    # save raw data
    adata.raw = adata

    # get highly variable genes
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    # print('highly variable X shape:{}'.format(adata.X.shape))
    adata_X = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
    adata_X = sc.pp.log1p(adata_X)
    adata_X = sc.pp.scale(adata_X)
    # pca
    if is_pac:
        adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
        print('pca X shape:{}'.format(adata_X.shape))

    # get highly variable gene name
    temp = adata.var.highly_variable
    hv_gene = temp[temp == True]
    hv_gene = np.array(hv_gene.index)

    return adata_X, hv_gene


def get_graph(X_data,
              coor_data=None,
              k_X=0,
              k_C=10,
              weight=0.5,
              include_self_X=False,
              include_self_C=False,
              graphType='kneighbors_graph_XC',
              ):
    if graphType == "kneighbors_graph":
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(coor_data, n_neighbors=k_C, include_self=include_self_C)
        A = A.toarray()

    elif graphType == "kneighbors_graph_X":
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(X_data, n_neighbors=k_X, include_self=include_self_C)
        A = A.toarray()

    elif graphType == "kneighbors_graph_C":
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(coor_data, n_neighbors=k_C, include_self=include_self_C)
        A = A.toarray()

    elif graphType == "kneighbors_graph_XC":
        from sklearn.neighbors import kneighbors_graph
        A_X = kneighbors_graph(X_data, n_neighbors=k_X, include_self=include_self_X)
        A_C = kneighbors_graph(coor_data, n_neighbors=k_C, include_self=include_self_C)
        A = weight * A_X + (1 - weight) * A_C
        A = A.toarray()

    return A


def filter_unmark_genes(adata, hv_gene, cluster_name='louvain', method='wilcoxon', mk_gene_num=100):
    # Identify marker genes for each cluster
    sc.tl.rank_genes_groups(adata, cluster_name, method=method)
    marker_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])

    mk_gene = np.unique(np.array(marker_genes.iloc[:mk_gene_num, ]).reshape(1, -1))
    # print("marker gene number:", len(mk_gene))
    unmark_gene = np.setdiff1d(hv_gene, mk_gene)
    # print("unmark gene number:", len(unmark_gene))
    result = np.isin(hv_gene, unmark_gene)
    # unmarker genes are True
    return result

