# train.py

import os

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
# Choose which model to train:
# from sod_model import SODNetBaseline as SODNet
from sod_model import SODNetImproved as SODNet  # improved version with Dropout


# ---------------------------
# IoU helpers
# ---------------------------

def soft_iou_from_logits(logits, targets, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft IoU (no threshold) – used inside the loss because it's differentiable.

    logits: [B, 1, H, W] raw scores from the model
    targets: [B, 1, H, W] ground truth masks in [0,1]
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def hard_iou_from_logits(logits, targets, eps: float = 1e-6) -> torch.Tensor:
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


# (Optional) kept for compatibility if you want to use Dice in experiments / evaluate.py
def dice_loss_from_logits(logits, targets, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss from logits (not used in the main loss, but available for experiments).
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def main():
    # Fix seeds so runs are a bit more reproducible.
    torch.manual_seed(42)
    np.random.seed(42)

    # Pick GPU if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Make sure the models directory exists.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pth")

    # Get train / val / test loaders from the local PNG dataset.
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        limit=None,          # you can set e.g. 300 here for quicker tests
        num_workers=NUM_WORKERS,
    )

    # Create the model and move it to the selected device.
    model = SODNet().to(device)

    # Binary Cross-Entropy on logits.
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # AMP scaler (speeds up training on GPU).
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    patience = 5   # early stopping patience
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # -------------------------
        # Training
        # -------------------------
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

                # BCE part.
                loss_bce = bce_loss(logits, masks)

                # IoU parts: soft for loss, hard for logging.
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

        # -------------------------
        # Validation
        # -------------------------
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
        print(f"  Val loss:   {avg_val_loss:.4f} | Val IoU:   {avg_val_iou:.4f}")

        # -------------------------
        # Early stopping + save best model
        # -------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved new best model to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
