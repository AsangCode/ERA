# Colab-ready PyTorch script for MNIST
# Goals:
# - Validation (50k/10k split) -> ~99.4% acc (typical)
# - < 20k trainable params (prints exact)
# - <= 20 epochs (uses 18)
# - Uses BatchNorm, Dropout, 1x1 and 3x3 convs, MaxPool, GAP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
from tqdm import trange

# -----------------------
# Reproducibility & device
# -----------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 32         # smaller batch size for better generalization
epochs = 20            # use full epoch budget
lr = 0.002            # increased learning rate for smaller batches
weight_decay = 0.00015 # slightly increased weight decay
dropout_p = 0.12       # carefully tuned dropout
data_root = "./data"

# -----------------------
# Data transforms (augment + normalize)
# - small rotations/translations: improve generalization
# -----------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(12),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------
# Datasets / Loaders
# -----------------------
full_train = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
train_size = 50000
val_size = len(full_train) - train_size  # 10000
train_set, val_set = random_split(full_train, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(seed))
# Use no-augment transforms for validation
val_set.dataset.transform = test_transform

test_set = datasets.MNIST(data_root, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# Model architecture
# Design notes (mapping to lecture points):
# - 3x3 convs for feature extraction (receptive field stacking)
# - 1x1 conv used as cheap channel-mixer (increase representational power)
# - MaxPool placed after initial feature extraction (early spatial reduction)
# - BatchNorm after convs for stable training
# - Dropout before GAP to regularize
# - GAP (AdaptiveAvgPool2d(1)) + small FC -> low param count
# -----------------------
class MNISTSmallNet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1, bias=False)   # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(10)
        
        # Deeper features with residual-like skip
        self.conv2a = nn.Conv2d(10, 10, kernel_size=3, padding=1, bias=False)  # 28x28 -> 28x28
        self.bn2a = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, kernel_size=3, padding=1, bias=False) # 28x28 -> 28x28
        self.bn2b = nn.BatchNorm2d(20)
        
        # First reduction block with 1x1 transition
        self.conv2c = nn.Conv2d(20, 16, kernel_size=1, bias=False)  # channel reduction
        self.bn2c = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)   # 28 -> 14
        
        # Second feature block
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False) # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(24)
        
        # Final reduction block
        self.pool2 = nn.MaxPool2d(2, 2)   # 14 -> 7
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1, bias=False) # 7x7 -> 7x7
        self.bn4 = nn.BatchNorm2d(32)
        
        # Final feature mixing with 1x1 convs
        self.conv5 = nn.Conv2d(32, 24, kernel_size=1, bias=False)  # channel reduction
        self.bn5 = nn.BatchNorm2d(24)
        
        self.dropout = nn.Dropout(dropout_p)
        
        # Global Average Pooling -> small FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, 10)
        
        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
    
    def forward(self, x):
        # Initial features
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Deeper features with skip connection
        identity = x
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = x + identity  # residual connection
        x = F.relu(self.bn2b(self.conv2b(x)))
        
        # First reduction with transition
        x = F.relu(self.bn2c(self.conv2c(x)))  # 1x1 conv transition
        x = self.dropout(x)  # early regularization
        x = self.pool1(x)
        
        # Second feature block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)  # mid-level regularization
        
        # Final reduction
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Final feature mixing
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)  # final regularization
        
        # Classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = MNISTSmallNet(dropout_p=dropout_p).to(device)

# -----------------------
# Print parameter count precisely (includes BN params)
# -----------------------
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print("Model:", model.__class__.__name__)
print("Params:", count_parameters(model))
# Ensure it's < 20k
if count_parameters(model) >= 20000:
    raise RuntimeError("Parameter budget exceeded! Adjust channels to be < 20k.")

# -----------------------
# Loss, optimizer, scheduler
# -----------------------
criterion = nn.CrossEntropyLoss()  # no label smoothing here, could add if desired
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# Cosine annealing across the limited epoch budget often helps (smooth decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# -----------------------
# Training & evaluation helpers
# -----------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

# -----------------------
# Training loop
# -----------------------
best_val = 0.0
best_epoch = -1
save_path = "mnist_smallnet_best.pth"

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    scheduler.step()   # cosine step each epoch

    if val_acc > best_val:
        best_val = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), save_path)
    print(f"Epoch {epoch:02d}/{epochs:02d}  train_loss={train_loss:.4f} train_acc={train_acc*100:.3f}%  "
          f"val_loss={val_loss:.4f} val_acc={val_acc*100:.3f}%")

print(f"Finished. Best val acc: {best_val*100:.4f}% at epoch {best_epoch}")

# -----------------------
# Evaluate best model on held-out MNIST test set (optional)
# -----------------------
model.load_state_dict(torch.load(save_path, map_location=device))
test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Official MNIST test acc: {test_acc*100:.4f}%  test_loss: {test_loss:.4f}")
