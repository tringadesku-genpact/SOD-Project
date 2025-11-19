import torch
from data_loader import get_dataloaders
from sod_model import SODNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        image_size=128,
    )

    batch = next(iter(train_loader))
    images = batch["images"]
    masks  = batch["masks"]

    print("Batch from train_loader:")
    print("  Images shape:", images.shape)
    print("  Masks shape:", masks.shape)

    model = SODNet().to(device)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("Model output shape:", outputs.shape)


if __name__ == "__main__":
    main()
