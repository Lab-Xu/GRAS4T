import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import math
from models import DGI
from utils import process
from tqdm import tqdm

from clustering_method.EfficientKernelGCSC import GCSC_Kernel


def gcsc_kernel_cluster(h, adj, regu_coef, gamma, ro, use_thrC=True):
    gcsc_k = GCSC_Kernel(adj, regu_coef=regu_coef, gamma=gamma, ro=ro,
                         save_affinity=False)

    mask = gcsc_k.fit(h, use_thrC=use_thrC)
    return mask


def norm_mask(mask, is_select=False, k=5, s_matrix=None):
    if is_select:
        if s_matrix is not None:
            print("Use the similarity matrix to help select nearest neighbor points.")
            mask = select_neighbor_v2(mask, s_matrix, k=k)
        else:
            print("Use subsapce select nearest neighbor points.")
            mask = select_neighbor(mask, k=k)
        # print("sum mask", np.sum(mask))
    mask = torch.FloatTensor(mask)
    row_sum = torch.sum(mask, 1)
    row_sum = row_sum.expand((mask.shape[1], mask.shape[0])).T
    mask = mask / row_sum

    return mask


def select_neighbor(mask, k=5):
    k = k + 1
    max_number = np.partition(mask, -k, axis=1)[:, -k]
    temp = np.ones((mask.shape[1], 1)) * max_number
    max_matrix = temp.T
    mask[mask < max_matrix] = 0

    return mask


def select_neighbor_v2(mask, s_matrix, k=5, ):
    k = k + 1
    compare_m = mask * s_matrix
    max_number = np.partition(compare_m, -k, axis=1)[:, -k]
    temp = np.ones((compare_m.shape[1], 1)) * max_number
    max_matrix = temp.T
    compare_m[compare_m < max_matrix] = 0
    indices = np.where(compare_m == 0)
    mask[indices] = 0

    return mask


def get_cluster_mask(y_pred: torch.FloatTensor, confidence=None):
    N = len(y_pred)
    mask = torch.zeros((N, N))
    for i in range(N):
        cluster_index_list = torch.where(y_pred == y_pred[i], torch.ones(N), torch.zeros(N))
        mask[i] = cluster_index_list
    if confidence:
        pass
    return mask.detach().to('cpu').numpy()


def get_cosine_similarity(H):
    H_temp = H.clone()
    H_temp = torch.squeeze(H_temp).detach().to('cpu').numpy()
    from sklearn.metrics.pairwise import cosine_similarity
    s = cosine_similarity(H_temp)
    return s


class DomainFeatureExtraction:
    def __init__(self, params,
                 features, aug_features1, aug_features2,
                 adj, aug_adj1, aug_adj2, seed=2022,
                 use_gpu=True):

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cuda"):
            print("USE CUDA!")
        elif self.device == torch.device("cpu"):
            print("USE CPU!")

        # params
        self.nb_nodes = params['nb_nodes']
        self.ft_size = params['ft_size']
        self.hid_units = params['hid_units']
        self.activation = params['activation']
        self.muti_act_1 = params['muti_act_1']
        self.muti_act_2 = params['muti_act_2']
        self.lr = params['lr']
        self.l2_coef = params['l2_coef']
        self.sparse = params['sparse']
        self.aug_type1 = params['aug_type1']
        self.aug_type2 = params['aug_type2']
        self.nb_epochs = params['nb_epochs']
        self.batch_size = params['batch_size']
        self.patience = params['patience']

        self.seed = seed

        # data
        self.features = features.to(self.device)
        self.aug_features1 = aug_features1.to(self.device)
        self.aug_features2 = aug_features2.to(self.device)

        self.adj = adj
        self.aug_adj1 = aug_adj1
        self.aug_adj2 = aug_adj2

        if self.sparse:
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(self.adj).to(self.device)
            self.sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(self.aug_adj1).to(self.device)
            self.sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(self.aug_adj2).to(self.device)
        else:
            self.adj = (self.adj + sp.eye(self.adj.shape[0])).todense()
            self.aug_adj1 = (self.aug_adj1 + sp.eye(self.aug_adj1.shape[0])).todense()
            self.aug_adj2 = (self.aug_adj2 + sp.eye(self.aug_adj2.shape[0])).todense()

        if not self.sparse:
            self.adj = torch.FloatTensor(self.adj[np.newaxis]).to(self.device)
            self.aug_adj1 = torch.FloatTensor(self.aug_adj1[np.newaxis]).to(self.device)
            self.aug_adj2 = torch.FloatTensor(self.aug_adj2[np.newaxis]).to(self.device)

        # model
        self.model = DGI(self.ft_size, self.hid_units,
                         self.activation, self.muti_act_1, self.muti_act_2).to(self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)

    def get_adj(self):
        if self.sparse:
            adj = (self.adj + sp.eye(self.adj.shape[0])).todense()
            adj = np.array(adj)
        else:
            adj = torch.squeeze(self.adj).detach().to('cpu').numpy()
        return adj

    def pretraining(self,
                    reconstruct=False,
                    weight_lc=1.0,
                    weight_recon=1.0,
                    nb_epochs_pretrain=100):
        cnt_wait = 0
        best = 1e9

        print('Begin pretraining...')
        for epoch in range(nb_epochs_pretrain):

            self.model.train()
            self.optimiser.zero_grad()
            idx = np.random.permutation(self.nb_nodes)
            shuf_fts = self.features[:, idx, :].to(self.device)

            lbl_1 = torch.ones(self.batch_size, self.nb_nodes).to(self.device)
            lbl_2 = torch.zeros(self.batch_size, self.nb_nodes).to(self.device)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            h_0, h_2, h_1, h_3 = self.model(self.features, shuf_fts, self.aug_features1, self.aug_features2,
                                            self.sp_adj if self.sparse else self.adj,
                                            self.sp_aug_adj1 if self.sparse else self.aug_adj1,
                                            self.sp_aug_adj2 if self.sparse else self.aug_adj2,
                                            self.sparse, aug_type1=self.aug_type1, aug_type2=self.aug_type2)

            h_0 = h_0.to(self.device)
            h_2 = h_2.to(self.device)
            h_1 = h_1.to(self.device)
            h_3 = h_3.to(self.device)

            if reconstruct:
                import torch.nn.functional as F
                X_ = self.model.decoder_head(h_0,
                                             self.sp_adj if self.sparse else self.adj,
                                             True)
                loss_feat = F.mse_loss(torch.squeeze(self.features), torch.squeeze(X_))

            loss = 0
            loss_node_global = self.model.node_global_loss(h_0, h_2, h_1, h_3, lbl)
            loss_node_global = weight_lc * loss_node_global
            loss += loss_node_global

            if reconstruct:
                loss_feat = weight_recon * loss_feat
                loss += loss_feat

            if (loss < best):
                best = loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                # print('Early stopping!')
                break

            loss.backward()
            self.optimiser.step()

        print("End pretraining...")

    def major_training(self,
                       condition_local_global=True,
                       condition_local_cluster=True,
                       reconstruct=True,
                       mid_cluster_method='gcsc',
                       use_thrC=True,
                       weight_lg=1.0,
                       weight_lc=1.0,
                       weight_recon=1.0,
                       regu_coef=1, gamma=0.2, ro=0.5,
                       interval_num=100,
                       is_select_neighbor=False,
                       select_neighbor_num=5,
                       use_recon_get_neighbor=False):
        cnt_wait = 0
        best = 1e9
        main_best = 1e9
        best_t = 0

        print('Begin train...')
        for epoch in range(self.nb_epochs):

            self.model.train()
            self.optimiser.zero_grad()
            idx = np.random.permutation(self.nb_nodes)
            shuf_fts = self.features[:, idx, :].to(self.device)

            lbl_1 = torch.ones(self.batch_size, self.nb_nodes).to(self.device)
            lbl_2 = torch.zeros(self.batch_size, self.nb_nodes).to(self.device)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            h_0, h_2, h_1, h_3 = self.model(self.features, shuf_fts, self.aug_features1, self.aug_features2,
                                            self.sp_adj if self.sparse else self.adj,
                                            self.sp_aug_adj1 if self.sparse else self.aug_adj1,
                                            self.sp_aug_adj2 if self.sparse else self.aug_adj2,
                                            self.sparse, aug_type1=self.aug_type1, aug_type2=self.aug_type2)

            h_0 = h_0.to(self.device)
            h_2 = h_2.to(self.device)
            h_1 = h_1.to(self.device)
            h_3 = h_3.to(self.device)

            if reconstruct:
                import torch.nn.functional as F
                X_ = self.model.decoder_head(h_0,
                                             self.sp_adj if self.sparse else self.adj,
                                             True)

            if condition_local_cluster:
                cluster_name = mid_cluster_method
                if epoch % interval_num == 0:
                    if mid_cluster_method == 'gcsc':
                        cluster_name = 'subspace'
                        adj = self.get_adj()
                        # print("adj type{}, adj shape{}".format(type(adj), adj.shape))
                        if use_recon_get_neighbor:
                            # print("use X_ to get neighbor")
                            use_X_ = torch.squeeze(X_).detach().to('cpu').numpy()
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=20, random_state=2023)
                            use_H = pca.fit_transform(use_X_.copy())
                        else:
                            # print("use h_0 to get neighbor")
                            use_H = torch.squeeze(h_0).detach().to('cpu').numpy()

                        mask = gcsc_kernel_cluster(use_H, adj, regu_coef=regu_coef, gamma=gamma, ro=ro,
                                                    use_thrC=use_thrC)
                    if cluster_name == "subspace":
                        mask = norm_mask(mask, is_select=is_select_neighbor, k=select_neighbor_num)
                    else:
                        mask = norm_mask(mask, is_select=is_select_neighbor, k=select_neighbor_num,
                                         s_matrix=get_cosine_similarity(h_0))
                    mask = mask.to(self.device)

                # print("mask type:{}, shape:{}".format(type(mask), mask.shape))
            else:
                mask = None

            # loss
            loss = 0
            loss_node_global = 0
            loss_node_cluster = 0
            nan_happen = False

            if condition_local_global:
                loss_node_global = self.model.node_global_loss(h_0, h_2, h_1, h_3, lbl)
                loss_node_global = weight_lg * loss_node_global
                loss += loss_node_global

            if reconstruct:
                loss_feat = F.mse_loss(torch.squeeze(self.features), torch.squeeze(X_))
                loss_feat = weight_recon * loss_feat
                loss += loss_feat

            if condition_local_cluster:
                loss_node_cluster = self.model.node_cluster_loss(h_0, h_2, h_1, h_3, lbl, msk=mask)
                loss_node_cluster = weight_lc * loss_node_cluster
                loss += loss_node_cluster

            if epoch % 100 == 0:
                if reconstruct:
                    print(
                        'Epoch {:0>3d} | Loss:[{:.4f}], loss_node_global:[{:.4f}], loss_node_cluster:[{:.4f}], loss_feat:[{:.4f}]'.format(
                            epoch,
                            loss.item(),
                            loss_node_global,
                            loss_node_cluster,
                            loss_feat))
                else:
                    print(
                        'Epoch {:0>3d} | Loss:[{:.4f}], loss_node_global:[{:.4f}], loss_node_cluster:[{:.4f}]'.format(
                            epoch,
                            loss.item(),
                            loss_node_global,
                            loss_node_cluster, ))

            if math.isnan(loss.item()):
                print('Appear nan, early stopping!')
                nan_happen = True
                break

            if (loss < best) | (loss_node_global < main_best):
                best = loss
                best_t = epoch
                cnt_wait = 0

                main_best = loss_node_global
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                break

            loss.backward()
            self.optimiser.step()

        print("End training...")

        print('Loading {}th epoch, best loss is {}'.format(best_t, best))
        return best, best_t, mask, nan_happen

    def get_embedding(self, use_decoder=False):

        embeds, X_, _ = self.model.embed(self.features, self.sp_adj if self.sparse else self.adj, self.sparse, None)
        if use_decoder:
            input_embeds = np.squeeze(X_.detach().to('cpu').numpy())
        else:
            input_embeds = np.squeeze(embeds.detach().to('cpu').numpy())

        return input_embeds
