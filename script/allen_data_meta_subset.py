import pandas as pd
import h5py

h5_path = "/home/users/z/zhouji/celltype_DNN/allen_data/expression_D_T.hdf5"

org_meta_path = "/home/users/z/zhouji/celltype_DNN/allen_data/metadata.csv"
out_meta_path = "/home/users/z/zhouji/celltype_DNN/allen_data/metadata_D_T.csv"

with h5py.File(h5_path, "r") as f:
    kept_cells = f["samples"][:]
    # HDF5 strings are normally stored as bytes
    if isinstance(kept_cells[0], bytes):
        kept_cells = [cell.decode('utf-8') for cell in kept_cells]

meta_df = pd.read_csv(org_meta_path)

barcode_col = "sample_name"

meta_df.set_index(barcode_col, inplace=True) #set sample_name as index

subset_meta = meta_df.loc[kept_cells]

subset_meta.reset_index(inplace=True) # return to original index

subset_meta.to_csv(out_meta_path, index=False)