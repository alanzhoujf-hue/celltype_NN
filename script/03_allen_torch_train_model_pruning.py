import numpy as np
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import sparse

from data_utils import load_adata_and_indices, make_dataloaders
from model import CellClassifier
from train_utils import train_one_epoch, evaluate

#load data
adata_path = "/home/users/z/zhouji/celltype_DNN/allen_data/ABA_adata_ctx_clean.h5ad"
trn_idx_path = "/home/users/z/zhouji/celltype_DNN/allen_data/trn_idx.npy"
val_idx_path = "/home/users/z/zhouji/celltype_DNN/allen_data/val_idx.npy"
tst_idx_path = "/home/users/z/zhouji/celltype_DNN/allen_data/tst_idx.npy"

stage1_best = "/home/users/z/zhouji/celltype_DNN/allen_data/model_stage1_best.pt" 

STAGE2_BEST_PATH = "/home/users/z/zhouji/celltype_DNN/allen_data/03_model_stage2_pruned_best.pt"
STAGE2_LAST_PATH = "/home/users/z/zhouji/celltype_DNN/allen_data/03_model_stage2_pruned_last.pt"
log_file_name = "/home/users/z/zhouji/celltype_DNN/allen_data/train_metrics_stage2_pruned.csv"

#load data
adata, trn_idx, val_idx, tst_idx = load_adata_and_indices(
    adata_path, trn_idx_path, val_idx_path, tst_idx_path
    )

trn_dl, val_dl, tst_dl = make_dataloaders(
    adata, trn_idx, val_idx, tst_idx, batch_size=64, normalize=True
    )

#device
device = "cuda" if torch.cuda.is_available() else "cpu"

#load 01 best checkpoint
stage1_ckpt = torch.load(stage1_best, map_location=device)

n_features = stage1_ckpt["n_features"]
n_classes = stage1_ckpt["n_classes"]

model = CellClassifier(
    n_features=n_features,
    n_classes=n_classes
    ).to(device)

model.load_state_dict(stage1_ckpt["model_state_dict"])

#pruning start here
model.compute_prune_mask(n=50)

#loss/optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#stage 2 loop
num_epochs = 20
best_val_acc = -1.0
best_epoch = -1

with open(log_file_name, "w") as f:
    f.write("epoch,train_loss,train_acc,val_loss, val_acc\n")
    
for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=trn_dl,
        criterion=criterion,
        optimizer=optimizer,
        device=device
        )
    
    val_loss, val_acc = evaluate(
        model=model,
        loader=val_dl,
        criterion=criterion,
        device=device
        )
    with open(log_file_name, "a") as f:
        f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(
            {"epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "n_features": n_features,
            "n_classes": n_classes
            },
            STAGE2_BEST_PATH
            )

torch.save({
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "n_features": n_features,
    "n_classes": n_classes
    },
    STAGE2_LAST_PATH
    )
