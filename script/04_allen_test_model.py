import numpy as np
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import CellDataset
from model import CellClassifier
from train_utils import evaluate

#path
ADATA_PATH = "/home/users/z/zhouji/celltype_DNN/allen_data/ABA_adata_ctx_clean.h5ad"
TST_idx_PATH = "/home/users/z/zhouji/celltype_DNN/allen_data/tst_idx.npy"

MODEL_best_PATH = "/home/users/z/zhouji/celltype_DNN/allen_data/03_model_stage2_pruned_best.pt"

#device
device = "cuda" if torch.cuda.is_available() else "cpu"

#load test data
adata = ad.read_h5ad(ADATA_PATH)
tst_idx = np.load(TST_idx_PATH)

tst_ds = CellDataset(adata, tst_idx, normalize=True)
tst_dl = DataLoader(tst_ds, batch_size=64, shuffle=False)

#load best model
chkp = torch.load(MODEL_best_PATH, map_location=device) #load model to device

n_features = chkp["n_features"]
n_classes = chkp["n_classes"]

model = CellClassifier(
    n_features=n_features,
    n_classes=n_classes
).to(device)

model.load_state_dict(chkp["model_state_dict"])

criterion = nn.CrossEntropyLoss() #loss

#test evaluation
test_loss, test_acc = evaluate(
    model = model,
    loader = tst_dl,
    criterion = criterion,
    device = device
)

result_path = "/home/users/z/zhouji/celltype_DNN/allen_data/test_result_stage2_best.txt"

with open(result_path, "w") as f:
    f.write(f"Total Loss = {test_loss:.4f}\n")
    f.write(f"Test Acc = {test_acc:.4f}\n")


