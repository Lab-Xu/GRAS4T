import numpy as np
import pandas as pd
import os
import anndata
import scanpy as sc


def build_her2st_data(path, name, image_feat_path=None):
    cnt_path = os.path.join(path, 'ST-cnts', f'{name}.tsv')
    df_cnt = pd.read_csv(cnt_path, sep='\t', index_col=0)

    pos_path = os.path.join(path, 'ST-spotfiles', f'{name}_selection.tsv')
    df_pos = pd.read_csv(pos_path, sep='\t')

    lbl_path = os.path.join(path, 'ST-pat/lbl', f'{name}_labeled_coordinates.tsv')
    df_lbl = pd.read_csv(lbl_path, sep='\t')
    df_lbl = df_lbl.dropna(axis=0, how='any')
    df_lbl.loc[df_lbl['label'] == 'undetermined', 'label'] = np.nan
    df_lbl['x'] = (df_lbl['x']+0.5).astype(np.int64)
    df_lbl['y'] = (df_lbl['y']+0.5).astype(np.int64)

    x = df_pos['x'].values
    y = df_pos['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i]))
    df_pos['id'] = ids

    x = df_lbl['x'].values
    y = df_lbl['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i]))
    df_lbl['id'] = ids

    meta_pos = df_cnt.join(df_pos.set_index('id'))
    meta_lbl = df_cnt.join(df_lbl.set_index('id'))

    adata = anndata.AnnData(df_cnt, dtype=np.int64)
    adata.obsm['spatial'] = np.floor(meta_pos[['pixel_x', 'pixel_y']].values).astype(int)

    adata.obs['Ground Truth'] = pd.Categorical(meta_lbl['label']).codes
    label = adata.obs['Ground Truth']
    adata = adata[label != -1]

    if image_feat_path is not None:
        image_feat = np.load(image_feat_path)
        adata.obsm["image_feat"] = image_feat

    return adata


def build_stereo_seq_data(path):
    counts_file = os.path.join(path, 'RNA_counts.tsv')
    coor_file = os.path.join(path, 'position.tsv')
    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    coor_df = pd.read_csv(coor_file, sep='\t')
    counts.columns = ['Spot_' + str(x) for x in counts.columns]
    coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
    coor_df = coor_df.loc[:, ['x', 'y']]
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    df_use_spot = pd.read_csv(path + '/used_barcodes.txt', sep='\t', header=None)
    use_spot_list = np.array(df_use_spot).reshape(-1)
    adata = adata[use_spot_list]
    coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
    adata.obsm["spatial"] = coor_df.to_numpy()

    return adata