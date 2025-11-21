import os
from glob import glob
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_FRACTION,
    VAL_FRACTION,
    SEED,
    NUM_WORKERS,
    get_paths,
)


class SaliencyDataset(Dataset):
    """Dataset for (image, mask) PNG pairs."""

    def __init__(self, image_paths: List[str], mask_paths: List[str], augment: bool):
        assert len(image_paths) == len(mask_paths), "Mismatch image/mask count."
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        )

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load_pair(idx)

        if self.augment:
            # random horizontal flip
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # random crop & resize back
            scale = 0.9
            w, h = img.size
            new_w, new_h = int(scale * w), int(scale * h)
            if new_w > 0 and new_h > 0:
                i, j, th, tw = T.RandomCrop.get_params(img, output_size=(new_h, new_w))
                img = TF.crop(img, i, j, th, tw)
                mask = TF.crop(mask, i, j, th, tw)

            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

            img = self.color_jitter(img)
        else:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        img_tensor = T.ToTensor()(img)  # [3,H,W], [0,1]

        mask_np = np.array(mask, dtype=np.float32) / 255.0
        # mask_np = (mask_np > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W]

        return {"images": img_tensor, "masks": mask_tensor}


def _get_image_mask_paths() -> Tuple[List[str], List[str]]:
    dataset_root, _ = get_paths()
    img_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")

    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(
            f"Expected 'images' and 'masks' folders inside {dataset_root}. "
            "Run prepare_dataset.py first or copy your dataset there."
        )

    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    if len(img_paths) == 0 or len(img_paths) != len(mask_paths):
        raise RuntimeError(
            f"Found {len(img_paths)} images and {len(mask_paths)} masks. "
            "They must be non-empty and equal."
        )

    print(f"Dataset root: {dataset_root}")
    print(f"Images: {len(img_paths)}, Masks: {len(mask_paths)}")
    return img_paths, mask_paths



def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
):
    img_paths, mask_paths = _get_image_mask_paths()
    n = len(img_paths)

    if limit is not None:
        n = min(n, limit)
        img_paths = img_paths[:n]
        mask_paths = mask_paths[:n]

    torch.manual_seed(SEED)
    indices = torch.randperm(n).tolist()

    train_end = int(TRAIN_FRACTION * n)
    val_end   = int((TRAIN_FRACTION + VAL_FRACTION) * n)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    def subset(paths, idxs):
        return [paths[i] for i in idxs]

    train_imgs = subset(img_paths, train_idx)
    train_masks = subset(mask_paths, train_idx)

    val_imgs = subset(img_paths, val_idx)
    val_masks = subset(mask_paths, val_idx)

    test_imgs = subset(img_paths, test_idx)
    test_masks = subset(mask_paths, test_idx)

    train_ds = SaliencyDataset(train_imgs, train_masks, augment=True)
    val_ds   = SaliencyDataset(val_imgs,   val_masks,   augment=False)
    test_ds  = SaliencyDataset(test_imgs,  test_masks,  augment=False)

    print(f"Total samples: {n}")
    print(f"Train: {len(train_ds)}")
    print(f"Val:   {len(val_ds)}")
    print(f"Test:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, _, _ = get_dataloaders(batch_size=4, limit=16)
    batch = next(iter(tl))
    print(batch["images"].shape, batch["masks"].shape)
