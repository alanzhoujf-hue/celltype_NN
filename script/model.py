import torch
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

