import anndata as ad 
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

#load anndata
ABA_adata_path = "/home/users/z/zhouji/celltype_DNN/allen_data/anndata_D_T_v2.h5ad"
ABA_adata = ad.read_h5ad(ABA_adata_path)

ABA_adata.obs["label_str"] = (
    ABA_adata.obs["class_label"].astype(str) + "|" +
    ABA_adata.obs["subclass_label"].astype(str)
)

#remove nan
bad = {"|", "nan|nan", "|nan", "nan|", ""}
ABA_adata = ABA_adata[~ABA_adata.obs["label_str"].isin(bad)].copy() #~ 取反 adata[mask]

#remove non-neocortex cells
neocortex_regions = [
    "VIS", "VISp", "VISl", "VISm",
    "AUD",
    "MOp", "MOs_FRP",
    "SSp", "SSs-GU-VISC-AIp",
    "ACA", "AI", "PL-ILA-ORB",
    "PTLp", "RSP"
]
ABA_adata_ctx = ABA_adata[ABA_adata.obs["region_label"].isin(neocortex_regions)].copy()

#remove low number cells
vc = ABA_adata_ctx.obs["label_str"].value_counts()
keep_label_str = vc[vc >= 100].index
ABA_adata_ctx = ABA_adata_ctx[ABA_adata_ctx.obs["label_str"].isin(keep_label_str)].copy()

#make torch data
class_names = sorted(ABA_adata_ctx.obs["label_str"].unique().tolist()) #a list
label_to_idx = {c: i for i, c in enumerate(class_names)} #enumerate(class_names) return an interator of (index, value) pairs
idx_to_label = {i: c for c, i in label_to_idx.items()}

ABA_adata_ctx.obs["label_idx"] = ABA_adata_ctx.obs["label_str"].map(label_to_idx).astype(int)

#make train/val/test sets
from sklearn.model_selection import train_test_split
y = ABA_adata_ctx.obs["label_idx"].to_numpy()
all_idx = np.arange(ABA_adata_ctx.n_obs)

#slice 50%  train, 50% tmp
trn_idx, tmp_idx = train_test_split(
    all_idx,
    test_size = 0.5,
    random_state = 12345,
    stratify = y
)

#split 50% tmp in to 25% val 25% test
val_idx, tst_idx = train_test_split(
    tmp_idx,
    test_size = 0.5,
    random_state = 12345,
    stratify = y[tmp_idx]
)

ABA_adata_ctx.write("/home/users/z/zhouji/celltype_DNN/allen_data/ABA_adata_ctx_clean.h5ad")
np.save("/home/users/z/zhouji/celltype_DNN/allen_data/trn_idx.npy", trn_idx)
np.save("/home/users/z/zhouji/celltype_DNN/allen_data/val_idx.npy", val_idx)
np.save("/home/users/z/zhouji/celltype_DNN/allen_data/tst_idx.npy", tst_idx)





