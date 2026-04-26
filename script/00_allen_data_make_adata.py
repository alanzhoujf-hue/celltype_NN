import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

#file paths
h5_path = "/home/users/z/zhouji/celltype_DNN/allen_data/expression.hdf5"
meta_path = "/home/users/z/zhouji/celltype_DNN/allen_data/metadata.csv"
tsne_path = "/home/users/z/zhouji/celltype_DNN/allen_data/coordinate.csv"

out_h5_path = "/home/users/z/zhouji/celltype_DNN/allen_data/expression_D_T_v2.hdf5"
out_meta_path = "/home/users/z/zhouji/celltype_DNN/allen_data/metadata_D_T_v2.csv"
out_tsne_path = "/home/users/z/zhouji/celltype_DNN/allen_data/coordinate_D_T_v2.csv"

barcode_col = "sample_name"
target_cells = 100000
chunk_size = 10000

meta_df = pd.read_csv(meta_path)

# Random sampling (random_state is ensuring reproducible results)
sampled_meta = meta_df.sample(n=target_cells, random_state=42)

# Convert the sampled cell barcodes into a Python 'Set' for ultra-fast lookup later
target_barcodes_set = set(sampled_meta[barcode_col].tolist())

# load ctx, locate cells
with h5py.File(h5_path, "r") as f:
    all_h5_samples = f["data"]["samples"][:]
    if isinstance(all_h5_samples[0], bytes):
        all_h5_samples = [s.decode('utf-8') for s in all_h5_samples]

n_total_h5_cells = len(all_h5_samples) #all cells

#create mask
mask = np.zeros(n_total_h5_cells, dtype=bool)
kept_barcodes_ordered = []

# Check each cell: if the HDF5 cell is in our 'sampled list', mark it as True
for i, cell_name in enumerate(all_h5_samples):
    if cell_name in target_barcodes_set:
        mask[i] = True
        kept_barcodes_ordered.append(cell_name)

#align metadata
sampled_meta.set_index(barcode_col, inplace=True)
#Because HDF5 is extracted sequentially, our Metadata must follow this exact order
final_meta = sampled_meta.loc[kept_barcodes_ordered].reset_index()
final_meta.to_csv(out_meta_path, index=False)

#subset tsne.csv
tsne_df = pd.read_csv(tsne_path)
tsne_df.set_index("sample_name", inplace = True)
tsne_sub = tsne_df.loc[kept_barcodes_ordered].reset_index()
tsne_sub.to_csv(out_tsne_path, index=False)

#subset expression matrix in batches
with h5py.File(h5_path, "r") as f_in, h5py.File(out_h5_path, "w") as f_out:
    counts_in = f_in["data"]["counts"]
    n_genes = counts_in.shape[0]
    n_kept_cells = len(kept_barcodes_ordered)

    #create the new dataset in the tranposed shape (Cell, Genes)
    counts_out = f_out.create_dataset(
        "expression_matrix",
        shape = (n_kept_cells, n_genes),
        dtype = counts_in.dtype,
        chunks = (2048, n_genes),
        compression = "lzf"
    )

    out_idx = 0
    for i in tqdm(range(0, n_total_h5_cells, chunk_size)):
        end = min(i+chunk_size, n_total_h5_cells)
        batch_mask = mask[i:end]
        cells_to_keep_in_batch = np.sum(batch_mask)

        if cells_to_keep_in_batch == 0:
            continue

        block = counts_in[:, i:end]
        filtered_block = block[:, batch_mask]
        counts_out[out_idx:out_idx+cells_to_keep_in_batch, :] = filtered_block.T
        out_idx = out_idx + cells_to_keep_in_batch

    f_out.create_dataset("genes", data=f_in["data"]["gene"][:])
    # Encode Unicode strings back to bytes for HDF5 compatibility
    encoded_samples = [s.encode('utf-8') for s in kept_barcodes_ordered]
    f_out.create_dataset("samples", data=np.array(encoded_samples))


#anndata assembly
import anndata as ad 
from scipy.sparse import csr_matrix

h5_path = "/home/users/z/zhouji/celltype_DNN/allen_data/expression_D_T_v2.hdf5"
meta_path = "/home/users/z/zhouji/celltype_DNN/allen_data/metadata_D_T_v2.csv"
tsne_path = "/home/users/z/zhouji/celltype_DNN/allen_data/coordinate_D_T_v2.csv"

#out anndata
out_h5ad_path = "/home/users/z/zhouji/celltype_DNN/allen_data/anndata_D_T_v2.h5ad"

obs_df = pd.read_csv(meta_path)
# AnnData requires the row names (index) of 'obs' to be the cell barcodes
obs_df.set_index("sample_name", inplace=True)

t_df = pd.read_csv(tsne_path)
#drop the sample_name column
tsne_coords = t_df.drop(columns="sample_name").to_numpy() #turn table in to numpy array (which was required by anndata)

with h5py.File(h5_path, "r") as f:
    X_dense = f["expression_matrix"][:]
    genes = f["genes"][:]
    if isinstance(genes[0], bytes):
        genes = [g.decode('utf-8') for g in genes]

var_df = pd.DataFrame(index=genes)
X_sparse = csr_matrix(X_dense)

adata = ad.AnnData(
    X=X_sparse,
    obs=obs_df,
    var=var_df
)

adata.obsm["X_coords"] = tsne_coords # has to be purely numbers (np array)

adata.write_h5ad(out_h5ad_path)