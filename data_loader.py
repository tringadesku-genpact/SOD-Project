import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import deeplake
import random


def get_dataloaders(batch_size: int = 8, image_size: int = 128):
    """
    Loads the ECSSD dataset from Deep Lake, splits it into train/val/test,
    applies preprocessing (resize + normalization) and some basic
    augmentations for the training set, and returns PyTorch DataLoaders.
    """

    # ----------------------------
    # Load the ECSSD dataset
    # ----------------------------
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
    # Base transforms for val/test
    # ----------------------------

    # Simple resize + ToTensor for images (no augmentation).
    val_image_tform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    def val_mask_tform(x):
        if x.ndim == 3:
            x = x[..., 0]

        pil = Image.fromarray(x)
        pil = pil.resize((image_size, image_size), resample=Image.NEAREST)

        mask_np = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

    # ----------------------------
    # Train-time joint transform (image + mask together)
    # ----------------------------

    color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def train_transform(sample):
        """
        Joint transform so that image and mask get the same geometric changes.
        Applies:
        - resize
        - random horizontal flip
        - random crop + resize back
        - brightness/contrast jitter (image only)
        """

        image = sample["images"]
        mask  = sample["masks"]

        # Convert to PIL images.
        img = Image.fromarray(image)
        if mask.ndim == 3:
            mask_arr = mask[..., 0]
        else:
            mask_arr = mask
        mask_img = Image.fromarray(mask_arr)

        # 1) Resize both to (image_size, image_size).
        img = img.resize((image_size, image_size), Image.BILINEAR)
        mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)

        # 2) Random horizontal flip (same flip for image and mask).
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask_img = TF.hflip(mask_img)

        # 3) Random crop (slightly smaller patch) then resize back.
        # This simulates random zoom/crop.
        crop_size = int(image_size * 0.9)  # e.g. 90% of the size
        i, j, h, w = T.RandomCrop.get_params(img, output_size=(crop_size, crop_size))

        img = TF.crop(img, i, j, h, w)
        mask_img = TF.crop(mask_img, i, j, h, w)

        img = img.resize((image_size, image_size), Image.BILINEAR)
        mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)

        # 4) Brightness/contrast jitter only on the image.
        img = color_jitter(img)

        # 5) Convert to tensors.
        img_tensor = T.ToTensor()(img)  # [3, H, W], in [0,1]

        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        return {
            "images": img_tensor,
            "masks": mask_tensor,
        }

    # ----------------------------
    # Build PyTorch DataLoaders via Deep Lake
    # ----------------------------

    # Train loader uses the joint train_transform so both image & mask
    # get the same random horizontal flip and crop.
    train_loader = train_ds.pytorch(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        tensors=["images", "masks"],
        transform=train_transform,
    )

    # Val / test loaders keep things deterministic (no augmentation).
    val_loader = val_ds.pytorch(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        tensors=["images", "masks"],
        transform={
            "images": val_image_tform,
            "masks":  val_mask_tform,
        },
    )

    test_loader = test_ds.pytorch(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        tensors=["images", "masks"],
        transform={
            "images": val_image_tform,
            "masks":  val_mask_tform,
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
