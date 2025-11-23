import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    MODELS_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
)
from data_loader import get_dataloaders

# Give this run a name so you don't overwrite others
# EXPERIMENT_NAME = "best_model"
EXPERIMENT_NAME = "sod_improved_dropout03_full"
# EXPERIMENT_NAME = "sod_improved_v2"


# Choose which model to use:
# from sod_model import SODNetBaseline as SODNet
from sod_model import SODNetImproved as SODNet
# from sod_model import SODNetImprovedV2 as SODNet


# IoU helpers
def soft_iou_from_logits(logits: torch.Tensor,
                         targets: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Soft IoU (no threshold) – used inside the loss (differentiable).
    logits:  [B, 1, H, W] raw scores from the model
    targets: [B, 1, H, W] ground truth masks in [0,1]
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def hard_iou_from_logits(logits: torch.Tensor,
                         targets: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Hard IoU (with threshold at 0.5) – used as a metric for logging.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def dice_loss_from_logits(logits: torch.Tensor,
                          targets: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss from logits (not used in main training loss, but
    kept for experiments or evaluation if you want to compare).
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()




def main():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Paths for model + log
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}.pth")
    history_csv = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_history.csv")

    # Data loaders (set limit=None for full dataset; smaller for quick tests)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        limit=None,          # use None for full data, e.g. 400 for quick experiments
        num_workers=NUM_WORKERS,
    )

    # Model
    model = SODNet().to(device)

    # Resume from checkpoint if exists
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}. Loading weights...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("→ Weights loaded, training will continue from this checkpoint.")
    else:
        print("→ No existing checkpoint found, training from scratch.")

    # Loss & optimizer (project requirement: Adam, lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Mixed precision scaler (only really helps on CUDA)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    patience = 5   # early stopping patience
    epochs_no_improve = 0

    # If history file doesn't exist, create it with header
    if not os.path.exists(history_csv):
        with open(history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "epoch",
                             "train_loss", "train_iou",
                             "val_loss", "val_iou"])

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for batch in tqdm(train_loader, desc="Train", leave=False):
            images = batch["images"].to(device)  # [B, 3, H, W]
            masks  = batch["masks"].to(device)   # [B, 1, H, W]

            optimizer.zero_grad()

            # Forward + loss in mixed precision (if on CUDA).
            with autocast(enabled=(device == "cuda")):
                logits = model(images)  # [B, 1, H, W]

                # BCE term.
                loss_bce = bce_loss(logits, masks)

                # IoU terms: soft for loss, hard for logging.
                iou_soft = soft_iou_from_logits(logits, masks)
                iou_hard = hard_iou_from_logits(logits, masks)

                # Project requirement:
                # Loss = BCE + 0.5 * (1 - IoU)
                loss = loss_bce + 0.5 * (1.0 - iou_soft)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_iou  += iou_hard.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou  = train_iou / len(train_loader)
        print(f"  Train loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                images = batch["images"].to(device)
                masks  = batch["masks"].to(device)

                logits = model(images)

                loss_bce = bce_loss(logits, masks)
                iou_soft = soft_iou_from_logits(logits, masks)
                iou_hard = hard_iou_from_logits(logits, masks)

                loss = loss_bce + 0.5 * (1.0 - iou_soft)

                val_loss += loss.item()
                val_iou  += iou_hard.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou  = val_iou / len(val_loader)
        print(f"  Val loss:   {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Save metrics to CSV
        with open(history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                epoch,
                f"{avg_train_loss:.6f}",
                f"{avg_train_iou:.6f}",
                f"{avg_val_loss:.6f}",
                f"{avg_val_iou:.6f}",
            ])

        # Early stopping + save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved new best model to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
