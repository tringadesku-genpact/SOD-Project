import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import deeplake


def get_dataloaders(batch_size: int = 8, image_size: int = 128):
    """
    Loads the ECSSD dataset from Deep Lake, splits it into train/val/test,
    applies basic preprocessing (resize + normalization), and returns
    PyTorch DataLoaders ready for training.
    """

    # ----------------------------
    # Load the ECSSD dataset
    # ----------------------------
    # Contains natural images and saliency masks.
    ds = deeplake.load("hub://activeloop/ecssd")

    # ----------------------------
    # Create shuffled indices and split (70/15/15)
    # ----------------------------
    n = len(ds)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    train_ds = ds[train_idx.tolist()]
    val_ds   = ds[val_idx.tolist()]
    test_ds  = ds[test_idx.tolist()]

    print("Total samples:", n)
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))

    # ----------------------------
    # Transforms for images and masks
    # ----------------------------

    # Convert images to PIL, resize to a fixed size, and turn into [0–1] tensors.
    image_tform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    # Masks are single-channel, so we keep them grayscale.
    # NEAREST interpolation avoids blurring the mask values.
    def mask_tform(x):
        if x.ndim == 3:
            x = x[..., 0]

        pil = Image.fromarray(x)
        pil = pil.resize((image_size, image_size), resample=Image.NEAREST)

        mask_np = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

    # ----------------------------
    # Build PyTorch DataLoaders via Deep Lake
    # ----------------------------

    train_loader = train_ds.pytorch(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        tensors=["images", "masks"],
        transform={
            "images": image_tform,
            "masks":  mask_tform,
        },
    )

    val_loader = val_ds.pytorch(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        tensors=["images", "masks"],
        transform={
            "images": image_tform,
            "masks":  mask_tform,
        },
    )

    test_loader = test_ds.pytorch(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        tensors=["images", "masks"],
        transform={
            "images": image_tform,
            "masks":  mask_tform,
        },
    )

    return train_loader, val_loader, test_loader


# If you run this file directly: quick sanity check
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    batch = next(iter(train_loader))
    images = batch["images"]
    masks  = batch["masks"]

    print(batch.keys())
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Images dtype:", images.dtype)
    print("Masks dtype:", masks.dtype)
