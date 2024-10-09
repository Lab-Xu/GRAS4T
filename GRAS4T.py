import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import random
from utils import process
import aug
import os
import scanpy as sc

import st.preprocessing as st_pp
from model_training import DomainFeatureExtraction
import anndata as ad
import tool.cluster_tool as ct
from tool.transform_tool import t_or_f


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm=None, cluster_key='mclust', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    if used_obsm:
        cluster_data = adata.obsm[used_obsm]
    else:
        cluster_data = adata.X
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(cluster_data), num_cluster, modelNames)

    mclust_res = np.array(res[-2])

    adata.obs[cluster_key] = mclust_res
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('int')
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
    return adata


def refine_label(adata, radius=150, key='label'):
    new_type = []
    old_type = adata.obs[key].values

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    import sklearn.neighbors
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    n_cell = indices.shape[0]
    for it in range(n_cell):
        neigh_type = [old_type[i] for i in indices[it]]
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(int(max_type))

    new_type = np.array(new_type, dtype=int)

    return new_type


def res_search_fixed_clus(adata, fixed_clus_count, clustering_name_, increment=0.02, seed=2023):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        if clustering_name_ == 'louvain':
            sc.tl.louvain(adata, resolution=res, random_state=seed)
        count_unique_leiden = len(pd.DataFrame(adata.obs[clustering_name_])[clustering_name_].unique())
        if count_unique_leiden <= fixed_clus_count:
            # print("predict cluster count:", count_unique_leiden)
            break
    print("best resolution:", res)
    return res


def run(args, adata,
        have_label=False,
        n_neighbors_=15,
        seed=2023, use_gpu=True,
        pretraining=False,
        nb_epochs_pretrain=100,
        l2_coef_=0.0, batch_size_=1, sparse_=True,
        condition_local_global=True,
        condition_local_cluster=True,
        reconstruct=True,
        hid_units_=[64],
        activation='prelu',
        muti_act_1='origin',
        muti_act_2='prelu',
        mid_cluster_method='gcsc',
        regu_coef=1, gamma=0.2,
        pca_embed=False,
        radius=150,
        use_decoder_emb=False,
        use_pca_emb=False,
        default_resolution=1.5,
        ):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not torch.cuda.is_available():
        use_gpu = False

    ############### GRAS4T ###############

    try:
        num_class = int(args.num_cluster)
    except:
        print("number of classes not provided")
        num_class = None

    #### preprocessing ####
    X_input, _ = st_pp.get_X(adata, n_top_genes=args.n_top_genes)
    coor_data_input = adata.obsm["spatial"]

    if have_label:
        if (args.platform=='10X') or (args.platform=='STARmap')or(args.platform == 'MERFISH'):
            data_truth = np.array(adata.obs['Ground Truth'])
            map_dict = {}
            for y_index, t in enumerate(set(data_truth)):
                map_dict[t] = y_index
            y_input = np.array([map_dict[i] for i in data_truth])

            if num_class is None:
                num_class = len([i for i in map_dict if isinstance(i, str)])
            else:
                pass
        else:
            y_input = np.array(adata.obs['Ground Truth'])
            if num_class is None:
                num_class = len(np.unique(y_input))
            else:
                pass

    print("X_input type:{}, shape:{}".format(type(X_input), X_input.shape))
    print("class num:{}".format(num_class))

    # construct A

    A = st_pp.get_graph(X_data=X_input,
                        coor_data=coor_data_input,
                        k_X=args.k_X_,
                        k_C=args.k_C_,
                        weight=args.weight_)

    A = (A + A.T) / 2

    print("A type:{}, shape:{}".format(type(A), A.shape))

    print("--" * 25)

    #### input setting ####
    features = np.matrix(X_input)
    nb_nodes = features.shape[0]  # node number
    ft_size = features.shape[1]  # node features dim
    features = torch.FloatTensor(features[np.newaxis])

    temp = np.matrix(A)
    adj = sp.csr_matrix(temp)

    #### graph augmentation ####

    aug_type1 = args.aug1
    aug_type2 = args.aug2
    drop_percent1 = args.drop_percent1
    drop_percent2 = args.drop_percent2

    print("Begin Aug1:[{}]".format(aug_type1))
    if aug_type1 == 'edge':

        aug_features1 = features
        aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent1)  # random drop edges

    elif aug_type1 == 'mask':

        aug_features1 = aug.aug_random_mask(features, drop_percent=drop_percent1)
        aug_adj1 = adj

    elif aug_type1 == 'HS_image':

        aug_features1 = features
        input_feature = adata.obsm["image_feat"]  # ["image_feat"], ["image_feat_pca"]
        aug_adj1 = aug.aug_HS_image(input_feature, adj, image_k=args.image_k, drop_percent=drop_percent1)

    else:
        assert False

    print("End Aug1:[{}]".format(aug_type1))

    print("\n")

    print("Begin Aug2:[{}]".format(aug_type2))
    if aug_type2 == 'edge':

        aug_features2 = features
        aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent2)  # random drop edges

    elif aug_type2 == 'mask':

        aug_features2 = aug.aug_random_mask(features, drop_percent=drop_percent2)
        aug_adj2 = adj

    elif aug_type2 == 'HS_image':

        aug_features2 = features
        input_feature = adata.obsm["image_feat"]
        aug_adj2 = aug.aug_HS_image(input_feature, adj, image_k=args.image_k, drop_percent=drop_percent2)

    else:
        assert False

    print("End Aug2:[{}]".format(aug_type2))

    print("--" * 25)

    ####### Domain Feature Extraction #######

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

    params = {}
    params['nb_nodes'] = nb_nodes
    params['ft_size'] = ft_size
    params['hid_units'] = hid_units_
    params['activation'] = activation
    params['muti_act_1'] = muti_act_1
    params['muti_act_2'] = muti_act_2
    params['lr'] = args.lr_
    params['l2_coef'] = l2_coef_
    params['sparse'] = sparse_
    params['aug_type1'] = aug_type1
    params['aug_type2'] = aug_type2
    params['nb_epochs'] = args.nb_epochs_
    params['batch_size'] = batch_size_
    params['patience'] = args.patience_

    gras4t_net = DomainFeatureExtraction(params,
                                     features, aug_features1, aug_features2,
                                     adj, aug_adj1, aug_adj2, seed=seed,
                                     use_gpu=use_gpu)

    #### pretraining ####

    if pretraining:
        print("--" * 25)

        gras4t_net.pretraining(nb_epochs_pretrain=nb_epochs_pretrain)
        input_embeds_pretraining = gras4t_net.get_embedding()
        adata.obsm['emb_pretrain'] = input_embeds_pretraining

        print("--" * 25)

    #### main train ####

    is_select_neighbor = t_or_f(args.is_select_neighbor)

    best, best_t, mask, nan_happen = gras4t_net.major_training(
        condition_local_global=condition_local_global,
        condition_local_cluster=condition_local_cluster,
        mid_cluster_method=mid_cluster_method,
        weight_lg=args.weight_lg,
        weight_lc=args.weight_lc,
        weight_recon=args.weight_recon,
        regu_coef=regu_coef, gamma=gamma,
        interval_num=args.interval_num,
        reconstruct=reconstruct,
        is_select_neighbor=is_select_neighbor,
        select_neighbor_num=args.select_neighbor,
        use_recon_get_neighbor=use_decoder_emb)

    input_embeds = gras4t_net.get_embedding(use_decoder=use_decoder_emb)

    if use_pca_emb:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20, random_state=seed)
        input_embeds = pca.fit_transform(input_embeds.copy())

    if use_decoder_emb:
        print('use pca clustering_input...')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20, random_state=seed)
        input_embeds = pca.fit_transform(input_embeds.copy())

    print("--" * 25)

    ####### Downstream Tasks #######

    print("clustering...")

    clustering_name_ = args.cluster_method

    if nan_happen:
        clustering_input = np.squeeze(features.detach().to('cpu').numpy())
        print('use original feature and louvain')
        clustering_name_ = 'louvain'
    else:
        clustering_input = input_embeds
        print('use embeding')

    if pca_embed:
        print('use pca clustering_input...')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20, random_state=seed)
        clustering_input = pca.fit_transform(clustering_input.copy())

    adata.obsm['emb'] = clustering_input

    if clustering_name_ == 'louvain':
        adataNew = ad.AnnData(clustering_input)
        sc.pp.neighbors(adataNew, n_neighbors=n_neighbors_, use_rep='X')
        if num_class is not None:
            eval_resolution = res_search_fixed_clus(adataNew, num_class, clustering_name_, seed=seed)
        else:
            eval_resolution = default_resolution
    elif clustering_name_ == 'mclust':
        if num_class is None:
            adataNew = ad.AnnData(clustering_input)
            sc.pp.neighbors(adataNew, n_neighbors=n_neighbors_, use_rep='X')
            sc.tl.louvain(adataNew, resolution=default_resolution, key_added='pre_cluster', random_state=seed)
            y_pre = np.array(adataNew.obs['pre_cluster'], dtype=int)
            num_class = len(np.unique(y_pre))

    if clustering_name_ == 'louvain':
        sc.tl.louvain(adataNew, resolution=eval_resolution, key_added=clustering_name_, random_state=seed)
        y_pre = np.array(adataNew.obs[clustering_name_], dtype=int)
    elif clustering_name_ == 'mclust':
        try:
            adata = mclust_R(adata, used_obsm='emb', num_cluster=num_class, random_seed=seed)
            y_pre = np.array(adata.obs[clustering_name_])
        except:
            print("error has occurred with mclust, using Louvain now!")
            clustering_name_ = 'louvain'
            adataNew = ad.AnnData(clustering_input)
            sc.pp.neighbors(adataNew, n_neighbors=n_neighbors_, use_rep='X')
            if num_class is None:
                eval_resolution = default_resolution
            else:
                eval_resolution = res_search_fixed_clus(adataNew, num_class, clustering_name_, seed=seed)
            sc.tl.louvain(adataNew, resolution=eval_resolution, key_added=clustering_name_, random_state=seed)
            y_pre = np.array(adataNew.obs[clustering_name_], dtype=int)

    if clustering_name_ == 'louvain':
        adata.obs[clustering_name_] = y_pre
        adata.obs[clustering_name_] = adata.obs[clustering_name_].astype('int')
        adata.obs[clustering_name_] = adata.obs[clustering_name_].astype('category')

    y_pre_post = refine_label(adata, key=clustering_name_, radius=radius)
    adata.obs['refine_label'] = y_pre_post
    adata.obs['refine_label'] = adata.obs[clustering_name_].astype('int')
    adata.obs['refine_label'] = adata.obs[clustering_name_].astype('category')

    if have_label:
        # clustering metrics
        ari_score, ami_score, nmi_score, fmi_score, aws_score = ct.cluster_evaluation(clustering_input, y_input, y_pre,
                                                                                      seed=seed)

        pre_class_num = len(np.unique(y_pre))

        print("origin | {}\n ARI:{:.4f}, AMI:{:.4f}, NMI:{:.4f}, FMI:{:.4f}".format(clustering_name_,
                                                                                    ari_score,
                                                                                    ami_score,
                                                                                    nmi_score,
                                                                                    fmi_score))
        print("predict number of class = ", pre_class_num)

        print("-" * 50)
        pre_class_num = len(np.unique(y_pre_post))

        ari_score_post, ami_score_post, nmi_score_post, fmi_score_post, aws_score_post = ct.cluster_evaluation(
            clustering_input, y_input, y_pre_post, seed=seed)
        print("refine | {}\n ARI:{:.4f}, AMI:{:.4f}, NMI:{:.4f}, FMI:{:.4f}".format(clustering_name_,
                                                                                    ari_score_post,
                                                                                    ami_score_post,
                                                                                    nmi_score_post,
                                                                                    fmi_score_post))
        print("predict number of class = ", pre_class_num)
        print("-" * 50)
    else:
        pre_class_num = len(np.unique(y_pre))
        print("predict number of class = ", pre_class_num)
        pre_class_num = len(np.unique(y_pre_post))
        print("refine predict number of class = ", pre_class_num)
        print("-" * 50)

