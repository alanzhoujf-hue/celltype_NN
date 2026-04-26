import torch

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

