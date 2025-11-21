import os
import numpy as np
from pathlib import Path
from PIL import Image
import deeplake
import shutil

from config import get_paths

def main():
    dataset_root, _ = get_paths()  # we only care about dataset_root here
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"

    # If already prepared, do nothing.
    if img_dir.is_dir() and mask_dir.is_dir() and any(img_dir.iterdir()):
        print(f"Dataset already found at: {dataset_root}")
        print(f"Images: {len(list(img_dir.iterdir()))}, Masks: {len(list(mask_dir.iterdir()))}")
        return

    # Fresh start
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ECSSD from Deep Lake...")
    ds = deeplake.load("hub://activeloop/ecssd")
    print("Samples:", len(ds))
    print("Exporting to:", dataset_root)

    for i, sample in enumerate(ds):
        img = sample["images"].numpy()
        mask = sample["masks"].numpy()

        # --- IMAGE ---
        img = np.asarray(img)

        # (H, W, 1) -> (H, W)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]

        # (H, W) -> (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # (H, W, 4) -> (H, W, 3)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]

        img = img.astype("uint8")
        img_pil = Image.fromarray(img)
        img_pil.save(img_dir / f"{i:04d}.jpg")

        # --- MASK ---
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype("uint8")
        mask_pil = Image.fromarray(mask)
        mask_pil.save(mask_dir / f"{i:04d}.png")

    print("Done.")
    print("Images:", len(list(img_dir.iterdir())))
    print("Masks:", len(list(mask_dir.iterdir())))


if __name__ == "__main__":
    main()
