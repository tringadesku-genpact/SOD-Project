import os
import torch
import torch.nn as nn
import numpy as np
from data_loader import get_dataloaders
from sod_model import SODNet
from config import IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, get_paths


# Path where the best model will be saved in Google Drive.
# Make sure in Colab you mounted drive at /content/drive before running this.
MODEL_PATH = "/content/drive/MyDrive/SOD/best_model.pth"


def batch_iou(preds, targets, eps=1e-6):
    """
    Computes IoU for a batch of predicted masks and target masks.
    Both inputs are expected in [0,1] with shape [B, 1, H, W].
    """
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def main():
    # A bit of reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    # Use GPU if available (Colab), else CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Resolve paths depending on environment (Colab vs local).
    dataset_root, model_dir = get_paths()
    print("Dataset root:", dataset_root)
    print("Model dir:", model_dir)

    # Make sure model directory exists before trying to save.
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pth")

    # Load train / val / test loaders from data_loader.py
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        dataset_root=dataset_root,
        limit=None,            # or e.g. 300 for a faster demo
        num_workers=NUM_WORKERS,
    )

    # Create the model and move it to the selected device.
    model = SODNet().to(device)

    # If a saved model already exists in Drive, load it and continue from there.
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}, loading weights...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded.")

    # Loss and optimizer.
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training settings.
    num_epochs = NUM_EPOCHS
    patience = 5  # for early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # -------------------------
        # Training
        # -------------------------
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["images"].to(device)  # [B, 3, 128, 128]
            masks  = batch["masks"].to(device)   # [B, 1, 128, 128]

            optimizer.zero_grad()

            # Forward pass.
            outputs = model(images)              # [B, 1, 128, 128]

            # BCE term
            loss_bce = bce_loss(outputs, masks)

            # IoU term (we want high IoU, so loss part is (1 - IoU))
            iou = batch_iou(outputs, masks)
            loss = loss_bce + 0.5 * (1.0 - iou)

            # Backprop + update weights.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"  Train loss: {avg_train_loss:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_iou_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                masks  = batch["masks"].to(device)

                outputs = model(images)

                loss_bce = bce_loss(outputs, masks)
                iou = batch_iou(outputs, masks)
                loss = loss_bce + 0.5 * (1.0 - iou)

                val_loss += loss.item()
                val_iou_total += iou.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_total / len(val_loader)

        print(f"  Val loss: {avg_val_loss:.4f}  |  Val IoU: {avg_val_iou:.4f}")

        # Early stopping + best model saving based on val loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            # Ensure directory exists (already done above, but safe to keep)
            os.makedirs(model_dir, exist_ok=True)

            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved new best model to {model_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= patience:
                print("\nEarly stopping triggered.")
                break

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
