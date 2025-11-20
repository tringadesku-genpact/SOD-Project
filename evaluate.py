import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from sod_model import SODNet
from config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, get_paths


def batch_iou(preds, targets, eps=1e-6):
    """
    Same IoU as in train.py.
    Expects tensors in [0,1], shape [B, 1, H, W].
    """
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def evaluate(model, test_loader, device):
    """
    Runs the model on the whole test set and prints average loss + IoU.
    """
    model.eval()
    bce_loss = nn.BCELoss()

    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            iou = batch_iou(outputs, masks)

            loss = loss_bce + 0.5 * (1.0 - iou)

            total_loss += loss.item()
            total_iou += iou.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_iou = total_iou / max(1, num_batches)

    print(f"\nTest results:")
    print(f"  Test loss: {avg_loss:.4f}")
    print(f"  Test IoU:  {avg_iou:.4f}")


def visualize_predictions(model, test_loader, device, num_samples=3, save_dir=None):
    """
    Takes a few samples from the test set and shows:
      - original image
      - ground truth mask
      - predicted mask

    If save_dir is given, saves the figures there instead of just showing them.
    """
    model.eval()

    # Just grab the first batch
    batch = next(iter(test_loader))
    images = batch["images"].to(device)
    masks = batch["masks"].to(device)

    with torch.no_grad():
        preds = model(images)  # [B, 1, H, W]

    # Move to CPU and numpy for plotting
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # Simple threshold for visualization
    preds_bin = (preds_np > 0.5).astype(np.float32)

    # Make sure we don't ask for more samples than we have
    num_samples = min(num_samples, images_np.shape[0])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        img = images_np[i].transpose(1, 2, 0)           # [C,H,W] -> [H,W,C]
        mask = masks_np[i, 0]                           # [1,H,W] -> [H,W]
        pred = preds_bin[i, 0]                          # [1,H,W] -> [H,W]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction (thr=0.5)")
        axes[2].axis("off")

        plt.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"sample_{i}.png"
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved {out_path}")
        else:
            plt.show()


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Resolve paths (Colab vs local)
    dataset_root, model_dir = get_paths()
    model_path = os.path.join(model_dir, "best_model.pth")
    print("Dataset root:", dataset_root)
    print("Model path:", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Train the model first (python train.py)."
        )

    # Data: we only need test_loader here
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        dataset_root=dataset_root,
        limit=None,          # or smaller for quick debug
        num_workers=NUM_WORKERS,
    )

    # Model
    model = SODNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model weights.")

    # 1) Quantitative metrics
    evaluate(model, test_loader, device)

    # 2) A few qualitative examples
    visualize_predictions(
        model,
        test_loader,
        device,
        num_samples=3,
        save_dir="outputs"   # will create /outputs and save images there
    )


if __name__ == "__main__":
    main()
