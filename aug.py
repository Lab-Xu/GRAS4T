import torch
import copy
import random
import scipy.sparse as sp
import numpy as np
import st.preprocessing as st_pp


def main():
    pass


def replace_zero_with_one(adj):
    temp = copy.deepcopy(adj)
    control_adj = copy.deepcopy(adj)
    control_adj[temp != 0] = 0.0
    control_adj[temp == 0] = 1.0
    return control_adj


def aug_HS_image(input_feature, input_adj, image_k=50, drop_percent=0.2):
    from sklearn.neighbors import kneighbors_graph
    A_image = kneighbors_graph(input_feature, n_neighbors=image_k, include_self=True).toarray()
    A_image = (A_image + A_image.T) / 2

    percent = drop_percent / 2

    aug_adj = copy.deepcopy(input_adj.todense().tolist())
    control_adj = copy.deepcopy(np.array(input_adj.todense()))

    adj_2 = replace_zero_with_one(control_adj)
    add_matrix = adj_2 * A_image
    adj_3 = replace_zero_with_one(A_image)
    drop_matrix = control_adj * adj_3

    row_one, col_one = add_matrix.nonzero()
    row_zero, col_zero = drop_matrix.nonzero()

    # add
    index_one = [(row_one[i], col_one[i]) for i in range(len(row_one))]

    edge_num_one = int(len(index_one) / 2)
    add_num = int(edge_num_one * percent / 2)
    print("added edge number:", add_num)

    if add_num >= 1:

        edge_one_idx = [i for i in range(edge_num_one)]
        add_idx = random.sample(edge_one_idx, add_num)

        for i in add_idx:
            aug_adj[index_one[i][0]][index_one[i][1]] = 1
            aug_adj[index_one[i][1]][index_one[i][0]] = 1
    else:
        print("nothing to modify!")

    # drop

    index_zero = [(row_zero[i], col_zero[i]) for i in range(len(row_zero))]

    edge_num_zero = int(len(index_zero) / 2)  # 9228 / 2
    drop_num = int(edge_num_zero * percent / 2)
    print("droped edge number:", drop_num)

    if drop_num >= 1:
        edge_zero_idx = [i for i in range(edge_num_zero)]
        add_idx = random.sample(edge_zero_idx, drop_num)

        for i in add_idx:
            aug_adj[index_zero[i][0]][index_zero[i][1]] = 0
            aug_adj[index_zero[i][1]][index_zero[i][0]] = 0
    else:
        print("nothing to modify!")

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)

    return aug_adj


def aug_unmarked_mask(adata, input_feature, hv_gene, cluster_name,
                      drop_percent=0.2, mk_gene_num=100):
    '''
    unmarked gene mask.
    '''

    aug_feature = input_feature.squeeze(0)
    unmarker_genes = st_pp.filter_unmark_genes(adata, hv_gene, cluster_name=cluster_name, mk_gene_num=mk_gene_num)
    print("unmarker genes number:", np.sum(unmarker_genes))

    # bluid mask tensor
    mask1 = torch.zeros(aug_feature.shape[0], aug_feature.shape[1])
    mask1[:, unmarker_genes] = 1

    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    print("masked row number:", mask_num)

    mask2 = torch.zeros(aug_feature.shape[0], aug_feature.shape[1])
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    mask2[mask_idx, :] = 1

    mask = mask1 * mask2

    indices = torch.where(mask == 1)
    aug_feature[indices] = 0

    aug_feature = aug_feature.unsqueeze(0)

    # print("aug_feature type:{}, shape:{}".format(type(aug_feature), aug_feature.shape))
    return aug_feature


def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    print("masked row number:", mask_num)

    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        # print("index list:", index_list)
        # print("i:", i)
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    print("droped or added edge number:", add_drop_num)

    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]

    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


if __name__ == "__main__":
    main()
