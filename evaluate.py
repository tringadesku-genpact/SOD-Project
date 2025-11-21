import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import MODELS_DIR, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS
from data_loader import get_dataloaders

# Choose model version to evaluate:
# from sod_model import SODNetBaseline as SODNet
from sod_model import SODNetImproved as SODNet

from train import (
    soft_iou_from_logits, 
    hard_iou_from_logits
)


def evaluate_and_visualize(num_samples: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train the model first (python train.py)."
        )

    # Only need test loader
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        limit=None,
        num_workers=NUM_WORKERS,
    )

    model = SODNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights.")

    # Same loss as used in training
    bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    # --- Evaluate whole test set ---
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            masks  = batch["masks"].to(device)

            logits = model(images)

            loss_bce = bce_loss(logits, masks)
            iou_soft = soft_iou_from_logits(logits, masks)
            loss = loss_bce + 0.5 * (1.0 - iou_soft)

            total_loss += loss.item()
            total_iou  += hard_iou_from_logits(logits, masks).item()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_iou  = total_iou / max(1, n_batches)

    print("\nTest results:")
    print(f"  Test loss: {avg_loss:.4f}")
    print(f"  Test IoU:  {avg_iou:.4f}")

    # --- Save visualizations ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    batch = next(iter(test_loader))
    images = batch["images"].to(device)
    masks  = batch["masks"].to(device)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)

    images_np = images.cpu().numpy()
    masks_np  = masks.cpu().numpy()
    preds_np  = (probs.cpu().numpy() > 0.5).astype(float)

    num_samples = min(num_samples, len(images_np))

    for i in range(num_samples):
        img = images_np[i].transpose(1, 2, 0)
        mask = masks_np[i, 0]
        pred = preds_np[i, 0]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction (0.5 threshold)")
        axes[2].axis("off")

        fig.tight_layout()
        out_path = OUTPUTS_DIR / f"sample_{i}.png"
        fig.savefig(out_path)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    evaluate_and_visualize(num_samples=3)
