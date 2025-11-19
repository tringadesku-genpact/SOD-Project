import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # Two conv layers with BatchNorm + ReLU.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SODNet(nn.Module):
    """
    Simple encoder–decoder (U-Net style) model for saliency / segmentation.
    Input:  [B, 3, H, W]
    Output: [B, 1, H, W]  (saliency mask)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # skip + upsampled

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final 1-channel output (mask)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)        # [B, 64, H,   W]
        x2 = self.pool1(x1)      # [B, 64, H/2, W/2]

        x2 = self.enc2(x2)       # [B, 128, H/2, W/2]
        x3 = self.pool2(x2)      # [B, 128, H/4, W/4]

        x3 = self.enc3(x3)       # [B, 256, H/4, W/4]
        x4 = self.pool3(x3)      # [B, 256, H/8, W/8]

        # Bottleneck
        x5 = self.bottleneck(x4) # [B, 512, H/8, W/8]

        # Decoder
        u3 = self.up3(x5)        # [B, 256, H/4, W/4]
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)        # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)        # [B, 64, H, W]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)

        out = self.out_conv(u1)  # [B, 1, H, W]
        out = torch.sigmoid(out) # mask in [0,1]

        return out
