
import numpy as np
import anndata as ad
import torch
from torch.utils.data import Dataset
from scipy import sparse

ABA_adata_ctx = ad.read_h5ad("/home/users/z/zhouji/celltype_DNN/allen_data/ABA_adata_ctx_clean.h5ad")
trn_idx = np.load("/home/users/z/zhouji/celltype_DNN/allen_data/trn_idx.npy")
val_idx = np.load("/home/users/z/zhouji/celltype_DNN/allen_data/val_idx.npy")
tst_idx = np.load("/home/users/z/zhouji/celltype_DNN/allen_data/tst_idx.npy")

#create class named CellDataset

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


trn_ds = CellDataset(ABA_adata_ctx, trn_idx, normalize=True) 
val_ds = CellDataset(ABA_adata_ctx, val_idx, normalize=True)
tst_ds = CellDataset(ABA_adata_ctx, tst_idx, normalize=True)


#load data
from torch.utils.data import DataLoader
trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
tst_dl = DataLoader(tst_ds, batch_size=64, shuffle=False)

#model
import torch.nn as nn
import torch.nn.functional as F

class CellClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.dropout_p = 0.5

        self.layer1_linear = nn.Linear(n_features, 1024, bias = False) #first layer, linear 

        self.layer2 = nn.Sequential(
            nn.Hardtanh(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

        #pruning mask
        self.register_buffer(
            "prune_mask",
            torch.ones_like(self.layer1_linear.weight, dtype=torch.bool)
        )
    
    @torch.no_grad()
    def compute_prune_mask(self, n=50):
        """
        For each hidden unit (each row of the first-layer weight), 
        keep the strongest n positive and n negative weights.
        """
        w = self.layer1_linear.weight.data #shape [1024, n_features]

        smallest_n = torch.topk(w, k=n, dim=1, largest=False).values[:, -1:].clone()
        largest_n = torch.topk(w, k=n, dim=1, largest=True).values[:,-1:].clone()

        mask = (w <= smallest_n) | (w >= largest_n)
        self.prune_mask.copy_(mask)

    def forward(self, rpm):
        #pruning and normalizing for first layer
        log_rpm = torch.log1p(rpm)

        mask_weight = self.layer1_linear.weight * self.prune_mask

        norms = mask_weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        normalized_weight = mask_weight / norms

        #scaling factor
        sf = log_rpm @ normalized_weight.abs().T
        sf = sf.clamp_min(0.01)

        x = F.linear(log_rpm, normalized_weight)
        x = x/sf
        x = F.dropout(x, p=self.dropout_p, training = self.training)
        x = self.layer2(x)

        return x

#device
device = "cuda" if torch.cuda.is_available() else "cpu"

n_features = ABA_adata_ctx.n_vars
n_classes = ABA_adata_ctx.obs["label_idx"].nunique()

model = CellClassifier(
    n_features=n_features,
    n_classes=n_classes
).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

@torch.no_grad() #no backward()
def evaluate(model, loader, criterion, device="cpu"):
    model.eval()

    total_loss=0.0
    correct=0
    total=0
    

    for batch in loader:
        x=batch["x"].to(device)
        y=batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    
    return total_loss / len(loader), correct / total

#train_one_epoch
def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    
    return total_loss / len(loader), correct / total



#run model
import os

num_epochs = 50
log_file_name = "/home/users/z/zhouji/celltype_DNN/allen_data/train_metrics.csv"
best_model_path = "/home/users/z/zhouji/celltype_DNN/allen_data/model_stage1_best.pt"
last_model_path = "/home/users/z/zhouji/celltype_DNN/allen_data/model_stage1_last.pt"

best_val_acc = -1.0
best_epoch = -1

#make empty csv
with open(log_file_name, "w") as f:
    f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

for epoch in range (1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model = model,
        loader = trn_dl,
        criterion = criterion,
        optimizer = optimizer,
        device = device
    )

    val_loss, val_acc = evaluate(
        model = model,
        loader = val_dl,
        criterion = criterion,
        device = device
    )

    print(
        f"Epoch {epoch:03d} | "
        f"train loss = {train_loss:.4f} | "
        f"train acc = {train_acc:.4f} | "
        f"val loss = {val_loss:.4f} | "
        f"val acc = {val_acc:.4f} | "
    )
    
    with open(log_file_name, "a") as f:
        f.write(f"{epoch}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}\n")
    
    #save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch

        torch.save(
            {"epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "n_features": n_features,
            "n_classes": n_classes
            },
            best_model_path
        )

# save last checkpoint
torch.save(
    {"epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "n_features": n_features,
    "n_classes": n_classes 
    },
    last_model_path
)


