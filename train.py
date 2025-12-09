# TEMP PATCH for Hugging Face + PyTorch 2.6 dev/nighly
from transformers.utils import import_utils
import torch

if "dev" in torch.__version__ or torch.__version__.startswith("2.6"):
    import_utils.check_torch_load_is_safe.__defaults__ = (False,)

# train.py
import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset_loader1 import ProductDataset
from model import FusionModel
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50
BATCH_SIZE = 8
EPOCHS = 12
ACCUM_STEPS = 4
LR = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_PROPORTION = 0.05
BEST_MODEL_PATH = "best_fusion_model.pth"
TRAIN_CSV = "/home/dinesh/AmazonMLChallange1/train.csv"
IMAGE_FOLDER = "/home/dinesh/AmazonMLChallange1/train_images"

# Load data and split
df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.08, random_state=42)

train_dataset = ProductDataset(train_df, IMAGE_FOLDER, is_train=True)
val_dataset = ProductDataset(val_df, IMAGE_FOLDER, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model, criterion, optimizer
model = FusionModel().to(DEVICE)
criterion = nn.SmoothL1Loss()  # L1 near zero, robust to outliers
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Scheduler with warmup
total_steps = math.ceil(len(train_loader) / ACCUM_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_PROPORTION)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler()

# Freeze image encoder for first 2 epochs
def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

set_requires_grad(model.image_model, False)
print("Frozen image encoder for initial warmup.")

best_val_rmse = 1e9
no_improve = 0
EARLY_STOPPING_PATIENCE = 3

def rmse(preds, targets):
    return torch.sqrt(((preds - targets) ** 2).mean()).item()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, input_ids, attention_mask, struct_feat, price) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        struct_feat = struct_feat.to(DEVICE, non_blocking=True)
        # log1p transform target
        price = torch.log1p(price.to(DEVICE, non_blocking=True))

        with autocast():
            outputs = model(images, input_ids, attention_mask, struct_feat)
            loss = criterion(outputs, price)

        scaler.scale(loss / ACCUM_STEPS).backward()
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item()

        if (i + 1) % PRINT_EVERY == 0:
            print(f"Epoch {epoch+1} Step {i+1}/{len(train_loader)} - avg loss {(running_loss/(i+1)):.4f}")

    avg_train_loss = running_loss / len(train_loader)

    # Unfreeze image encoder after 2 epochs
    if epoch == 1:
        set_requires_grad(model.image_model, True)
        print("Unfroze image encoder for fine-tuning.")

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for (images, input_ids, attention_mask, struct_feat, price) in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            struct_feat = struct_feat.to(DEVICE, non_blocking=True)
            price = torch.log1p(price.to(DEVICE, non_blocking=True))
            with autocast():
                outputs = model(images, input_ids, attention_mask, struct_feat)
            val_preds.append(outputs.detach().cpu())
            val_targets.append(price.detach().cpu())
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_rmse = rmse(torch.expm1(val_preds), torch.expm1(val_targets))  # compute RMSE on original scale

    print(f"Epoch {epoch+1}/{EPOCHS} - TrainLoss: {avg_train_loss:.4f} Val_RMSE: {val_rmse:.4f}")

    # Save best
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved best model with val_rmse={best_val_rmse:.4f}")
        no_improve = 0
    else:
        no_improve += 1
        print(f"No improvement for {no_improve} epochs.")
        if no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training finished. Best val RMSE:", best_val_rmse)
