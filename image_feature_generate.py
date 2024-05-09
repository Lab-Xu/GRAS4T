import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import st.preprocessing as st_pp


data_slice = '151507'
cnnType_ = 'Resnet152'
crop_size = 50
target_size = 224

# create tile image path
save_path = "./Results"

# read data
input_dir = './data/10X/DLPFC/' + data_slice
adata = sc.read_visium(path=input_dir, count_file=data_slice + '_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

quality = 'hires'
library_id = list(adata.uns["spatial"].keys())[0]
scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"]
image_coor = adata.obsm["spatial"] * scale
adata.obs["imagecol"] = image_coor[:, 0]
adata.obs["imagerow"] = image_coor[:, 1]
adata.uns["spatial"][library_id]["use_quality"] = quality

image = adata.uns["spatial"][library_id]["images"][
    adata.uns["spatial"][library_id]["use_quality"]]

if image.dtype == np.float32 or image.dtype == np.float64:
    image = (image * 255).astype(np.uint8)
img_pillow = Image.fromarray(image)
tile_names = []

with tqdm(total=len(adata),
          desc="Tiling image",
          bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
        imagerow_down = imagerow - crop_size / 2
        imagerow_up = imagerow + crop_size / 2
        imagecol_left = imagecol - crop_size / 2
        imagecol_right = imagecol + crop_size / 2
        tile = img_pillow.crop(
            (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
        tile.resize((target_size, target_size))
        tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
        out_tile = Path(save_path) / (tile_name + ".png")
        tile_names.append(str(out_tile))
        tile.save(out_tile, "PNG")
        pbar.update(1)

adata.obs["slices_path"] = tile_names

adata = st_pp.image_feature(adata, pca_components=300, cnnType=cnnType_).extract_image_feat()
image_feat = adata.obsm["image_feat"]
# np.save("./data/10X/DLPFC/image_feature/{}-{}image_feat.npy".format(data_slice, cnnType_), image_feat)
