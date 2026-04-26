import h5py
import numpy as np
import pandas as pd
import anndata as ad

h5_path = "/home/users/z/zhouji/celltype_DNN/allen_data/expression.hdf5"

f = h5py.File(h5_path, "r")
list(f.keys())
f["data"]["counts"]
print(type(f["data"]))
print(list(f["data"].keys()))

with h5py.File(h5_path, "r") as f:
    g = f["data"]
    print("counts shape:", g["counts"].shape, g["counts"].dtype)
    print("gene shape:", g["gene"].shape, g["gene"].dtype)
    print("samples shape:", g["samples"].shape, g["samples"].dtype)
    print("shape value:", g["shape"][()])
    print("first 5 genes:", g["gene"][:5])
    print("first 5 samples:", g["samples"][:5])

from tqdm import tqdm

def downsample_and_transpose_hdf5(input_path, output_path, target_cells=100000, chunk_size=10000):
    """
    Randomly downsamples and transposes a massive HDF5 dataset 
    by processing it in memory-safe sequential batches.
    """
    print("Opening files...")
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        counts_in = f_in["data"]["counts"]
        n_genes, n_cells = counts_in.shape
        
        print(f"Original shape: {n_genes} genes, {n_cells} cells")
        
        if target_cells >= n_cells:
            raise ValueError("Target cells must be less than total cells.")
        
        # 1. Create a boolean mask for random sampling
        print(f"Randomly selecting {target_cells} cells...")
        mask = np.zeros(n_cells, dtype=bool)
        # Randomly choose indices without replacement
        chosen_indices = np.random.choice(n_cells, target_cells, replace=False)
        mask[chosen_indices] = True
        
        # 2. Create the new transposed and downsampled dataset in the output file
        counts_out = f_out.create_dataset(
            "expression_matrix",               
            shape=(target_cells, n_genes),     # PyTorch shape: (Cells, Genes)
            dtype=counts_in.dtype,             
            chunks=(2048, n_genes),            # Optimized for row-by-row PyTorch loading
            compression="lzf"                  
        )
        
        # Prepare to store the cell names/barcodes we actually keep
        samples_in = f_in["data"]["samples"]
        samples_out_list = []
        
        # 3. Read sequentially, filter in RAM, and write out
        print("Processing, filtering, and transposing in batches...")
        out_idx = 0 # Tracks our current row in the new file
        
        for i in tqdm(range(0, n_cells, chunk_size)):
            end = min(i + chunk_size, n_cells)
            
            # Get the mask slice for this specific batch
            batch_mask = mask[i:end]
            cells_to_keep_in_batch = np.sum(batch_mask)
            
            # If no cells were randomly chosen in this chunk, skip disk read entirely
            if cells_to_keep_in_batch == 0:
                continue
                
            # Read a safe block into RAM (e.g., 31,053 genes x 10,000 cells max)
            block = counts_in[:, i:end]
            
            # Filter the block in RAM (keep only the True columns)
            filtered_block = block[:, batch_mask]
            
            # Transpose and write to the output file
            counts_out[out_idx : out_idx + cells_to_keep_in_batch, :] = filtered_block.T
            
            # Save the sample names for the cells we kept
            samples_out_list.extend(samples_in[i:end][batch_mask])
            
            # Move our output index forward
            out_idx += cells_to_keep_in_batch
        
        # 4. Copy over metadata
        print("Writing metadata...")
        # FIXED: Directly read the gene array and write it to the new file
        f_out.create_dataset("genes", data=f_in["data"]["gene"][:])
        
        # Save the filtered cell names
        f_out.create_dataset("samples", data=np.array(samples_out_list))
        
    print(f"Done! Downsampled and transposed matrix saved to: {output_path}")

# --- Run the function ---
INPUT_FILE = "/home/users/z/zhouji/celltype_DNN/allen_data/expression.hdf5" 
OUTPUT_FILE = "/home/users/z/zhouji/celltype_DNN/allen_data/expression_D_T.hdf5"

# Set target_cells to whatever your RAM/GPU can comfortably handle during training
downsample_and_transpose_hdf5(INPUT_FILE, OUTPUT_FILE, target_cells=100000)