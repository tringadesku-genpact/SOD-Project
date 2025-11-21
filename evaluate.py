import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import MODELS_DIR, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS
from data_loader import get_dataloaders

# Choose model version to evaluate:
# from sod_model import SODNetBaseline as SODNet
# from sod_model import SODNetImproved as SODNet
from sod_model import SODNetImprovedV2 as SODNet

# Use the same experiment name as in train.py
from train import soft_iou_from_logits, hard_iou_from_logits  # reuse helpers
from train import EXPERIMENT_NAME


def compute_metrics_over_batch(logits: torch.Tensor,
                               targets: torch.Tensor,
                               bce_loss: nn.Module,
                               eps: float = 1e-6):
    """
    Computes loss + IoU + Precision + Recall + F1 + MAE for a batch.
    All metrics are returned as scalars (Python floats).
    """
    # BCE + IoU-based loss, same style as training (but no 0.5 factor here;
    # we'll construct final loss outside if we want).
    loss_bce = bce_loss(logits, targets)
    iou_soft = soft_iou_from_logits(logits, targets)

    # For metric IoU, use hard thresholded version
    iou_hard = hard_iou_from_logits(logits, targets)

    # Probabilities and binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Flatten for classification-style metrics
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # True positives, false positives, false negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    # Mean Absolute Error between probs and targets
    mae = torch.mean(torch.abs(probs - targets))

    # Final loss (project style): BCE + 0.5 * (1 - IoU_soft)
    loss = loss_bce + 0.5 * (1.0 - iou_soft)

    return (
        loss.item(),
        iou_hard.item(),
        precision.item(),
        recall.item(),
        f1.item(),
        mae.item(),
    )


def evaluate_and_visualize(num_samples: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Make sure outputs dir exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model path — must match train.py naming
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}.pth"
    print("Model path:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. "
            "Make sure you've trained and saved this experiment first."
        )

    # Only need test loader for evaluation
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        limit=None,
        num_workers=NUM_WORKERS,
    )

    # Model
    model = SODNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights.")

    # Loss function
    bce_loss = nn.BCEWithLogitsLoss()

    # ---- Aggregate metrics on full test set ----
    total_loss = 0.0
    total_iou  = 0.0
    total_prec = 0.0
    total_rec  = 0.0
    total_f1   = 0.0
    total_mae  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            masks  = batch["masks"].to(device)

            logits = model(images)

            loss, iou, prec, rec, f1, mae = compute_metrics_over_batch(
                logits, masks, bce_loss
            )

            total_loss += loss
            total_iou  += iou
            total_prec += prec
            total_rec  += rec
            total_f1   += f1
            total_mae  += mae
            n_batches  += 1

    if n_batches == 0:
        raise RuntimeError("Test loader is empty.")

    avg_loss = total_loss / n_batches
    avg_iou  = total_iou / n_batches
    avg_prec = total_prec / n_batches
    avg_rec  = total_rec / n_batches
    avg_f1   = total_f1 / n_batches
    avg_mae  = total_mae / n_batches

    print("\nTest results (averaged over test set):")
    print(f"  Test loss:       {avg_loss:.4f}")
    print(f"  Test IoU:        {avg_iou:.44f}")
    print(f"  Test Precision:  {avg_prec:.4f}")
    print(f"  Test Recall:     {avg_rec:.4f}")
    print(f"  Test F1-score:   {avg_f1:.4f}")
    print(f"  Test MAE:        {avg_mae:.4f}")

    # ---- Visualize some predictions ----
    # To avoid always using the first batch, pick a random batch index
    all_batches = list(test_loader)
    chosen_batch_idx = random.randint(0, len(all_batches) - 1)
    batch = all_batches[chosen_batch_idx]

    images = batch["images"].to(device)
    masks  = batch["masks"].to(device)

    with torch.no_grad():
        logits = model(images)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).float()

    images_np = images.cpu().numpy()
    masks_np  = masks.cpu().numpy()
    preds_np  = preds.cpu().numpy()

    num_samples = min(num_samples, images_np.shape[0])

    for i in range(num_samples):
        img  = images_np[i].transpose(1, 2, 0)          # [H, W, 3], in [0,1]
        mask = masks_np[i, 0]                           # [H, W]
        pred = preds_np[i, 0]                           # [H, W]

        # Overlay: tint prediction in red on top of the RGB image
        # Make sure everything is float in [0,1]
        img_clipped   = np.clip(img, 0.0, 1.0)
        pred_clipped  = np.clip(pred, 0.0, 1.0)

        pred_rgb = np.stack([pred_clipped, np.zeros_like(pred_clipped), np.zeros_like(pred_clipped)], axis=-1)
        overlay = (0.7 * img_clipped + 0.3 * pred_rgb)
        overlay = np.clip(overlay, 0.0, 1.0)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].imshow(img_clipped)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("GT saliency mask")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Predicted mask")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (pred + image)")
        axes[3].axis("off")

        fig.tight_layout()
        out_path = OUTPUTS_DIR / f"sample_{i}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    evaluate_and_visualize(num_samples=3)
