import numpy as np
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class CellDataset(Dataset):
    def __init__(self, adata, indices, normalize=True): #__init__ serves as the instrnctor method for classes
        self.adata = adata
        self.indices = np.asarray(indices)
        self.normalize = normalize

    def __len__(self): #will need later at Dataloader
        return len(self.indices)

    def __getitem__(self, i): #will need later at Dataloader
        idx = self.indices[i]
        x = self.adata.X[idx]

        if sparse.issparse(x):
            x = x.toarray().ravel()
        else:
            x = np.asarray(x).ravel()
        x = x.astype(np.float32)

        if self.normalize:
            libsize = max(x.sum(), 100.0)
            x = x / libsize * 1e4

        y = int(self.adata.obs["label_idx"].iloc[idx])

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long)
        }


def load_adata_and_indices(adata_path, trn_idx_path, val_idx_path, tst_idx_path):
    adata = ad.read_h5ad(adata_path)
    trn_idx = np.load(trn_idx_path)
    val_idx = np.load(val_idx_path)
    tst_idx = np.load(tst_idx_path)
    return adata, trn_idx, val_idx, tst_idx

def make_dataloaders(adata, trn_idx, val_idx, tst_idx, batch_size=64, normalize=True):
    trn_ds = CellDataset(adata, trn_idx, normalize=normalize)
    val_ds = CellDataset(adata, val_idx, normalize=normalize)
    tst_ds = CellDataset(adata, tst_idx, normalize=normalize)

    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, shuffle=False)
    
    return trn_dl, val_dl, tst_dl