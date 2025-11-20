import os
from glob import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ECSSDDataset(Dataset):
    """
    Simple Dataset wrapper around a local ECSSD-style folder:
      root/
        images/0000.jpg, 0001.jpg, ...
        masks/ 0000.png, 0001.png, ...

    It returns a dict with:
      "images": [3, H, W] float32 in [0,1]
      "masks":  [1, H, W] float32 in [0,1]
    """

    def __init__(
        self,
        img_paths,
        mask_paths,
        indices,
        image_size: int = 128,
        augment: bool = False,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.indices = indices
        self.image_size = image_size
        self.augment = augment

        # Light photometric augmentation for the image.
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map local index -> global file index.
        real_idx = self.indices[idx]
        img_path = self.img_paths[real_idx]
        mask_path = self.mask_paths[real_idx]

        # Open image (RGB) and mask (single channel).
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both to a fixed size.
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Optional augmentations only for the training split.
        if self.augment:
            # Horizontal flip (same flip for image and mask).
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Brightness/contrast jitter only on the image.
            img = self.color_jitter(img)

        # Convert to tensors.
        img_tensor = T.ToTensor()(img)  # [3, H, W], values in [0,1]

        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        return {
            "images": img_tensor,
            "masks": mask_tensor,
        }


def get_dataloaders(
    batch_size: int = 8,
    image_size: int = 128,
    dataset_root: str = "/content/dataset_ecssd",
    limit: int | None = None,
    num_workers: int = 2,
):
    """
    Builds train / val / test DataLoaders from a local ECSSD folder.

    - dataset_root should contain:
        images/*.jpg
        masks/*.png
    - limit can be used to only take the first N samples
      (handy for faster debug or demo).
    """

    img_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")

    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(
            f"Expected folders 'images' and 'masks' inside {dataset_root}. "
            "Make sure you exported ECSSD from Deep Lake first."
        )

    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    if len(img_paths) == 0 or len(img_paths) != len(mask_paths):
        raise RuntimeError(
            f"Found {len(img_paths)} images and {len(mask_paths)} masks. "
            "They should be non-empty and have the same length."
        )

    # Optional: only use a subset for faster training / demos.
    if limit is not None:
        img_paths = img_paths[:limit]
        mask_paths = mask_paths[:limit]

    n = len(img_paths)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n))

    print("Total samples:", n)
    print("Train:", len(train_indices))
    print("Val:", len(val_indices))
    print("Test:", len(test_indices))

    # Create dataset objects for each split.
    train_ds = ECSSDDataset(
        img_paths,
        mask_paths,
        indices=train_indices,
        image_size=image_size,
        augment=True,   # augmentations only on train
    )
    val_ds = ECSSDDataset(
        img_paths,
        mask_paths,
        indices=val_indices,
        image_size=image_size,
        augment=False,
    )
    test_ds = ECSSDDataset(
        img_paths,
        mask_paths,
        indices=test_indices,
        image_size=image_size,
        augment=False,
    )

    # Wrap them in DataLoaders.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# If you run this file directly: quick sanity check.
if __name__ == "__main__":
    loaders = get_dataloaders(limit=100)  # smaller subset just to test
    train_loader, val_loader, test_loader = loaders

    batch = next(iter(train_loader))
    images = batch["images"]
    masks = batch["masks"]

    print(batch.keys())
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Images dtype:", images.dtype)
    print("Masks dtype:", masks.dtype)
