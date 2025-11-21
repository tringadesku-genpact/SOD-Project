import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two conv layers + BatchNorm + ReLU.
    Optional dropout at the end of the block.
    """

    def __init__(self, in_ch: int, out_ch: int, p_dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

        # Only add dropout if p_dropout > 0
        if p_dropout > 0.0:
            # 2D dropout works on feature maps (channels, H, W)
            layers.append(nn.Dropout2d(p=p_dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SODNetBaseline(nn.Module):
    """
    Simple U-Net style encoder–decoder (baseline).
    Input:  [B, 3, H, W]
    Output: [B, 1, H, W] (logits; Sigmoid is applied in loss/metrics)
    """

    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # up4 + enc4

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # up3 + enc3

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # up2 + enc2

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # up1 + enc1

        # Final 1-channel output (logits)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)        # [B, 64, H,   W]
        x2 = self.pool1(x1)      # [B, 64, H/2, W/2]

        x2 = self.enc2(x2)       # [B, 128, H/2, W/2]
        x3 = self.pool2(x2)      # [B, 128, H/4, W/4]

        x3 = self.enc3(x3)       # [B, 256, H/4, W/4]
        x4 = self.pool3(x3)      # [B, 256, H/8, W/8]

        x4 = self.enc4(x4)       # [B, 512, H/8, W/8]
        x5 = self.pool4(x4)      # [B, 512, H/16, W/16]

        # Bottleneck
        x5 = self.bottleneck(x5)  # [B, 1024, H/16, W/16]

        # Decoder
        u4 = self.up4(x5)         # [B, 512, H/8, W/8]
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)         # [B, 256, H/4, W/4]
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)         # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)         # [B, 64, H, W]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)

        logits = self.out_conv(u1)  # [B, 1, H, W]
        return logits               # we apply Sigmoid outside


class SODNetImproved(nn.Module):
    """
    Improved U-Net style model:
    - Same overall structure as baseline
    - Adds Dropout2d in each DoubleConv block (p=0.3) to reduce overfitting
    """

    def __init__(self, p_dropout: float = 0.3):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = DoubleConv(3, 64, p_dropout=p_dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128, p_dropout=p_dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256, p_dropout=p_dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512, p_dropout=p_dropout)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(512, 1024, p_dropout=p_dropout)

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512, p_dropout=p_dropout)  # up4 + enc4

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, p_dropout=p_dropout)   # up3 + enc3

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, p_dropout=p_dropout)   # up2 + enc2

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, p_dropout=p_dropout)    # up1 + enc1

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool1(x1)

        x2 = self.enc2(x2)
        x3 = self.pool2(x2)

        x3 = self.enc3(x3)
        x4 = self.pool3(x3)

        x4 = self.enc4(x4)
        x5 = self.pool4(x4)

        # Bottleneck
        x5 = self.bottleneck(x5)

        # Decoder
        u4 = self.up4(x5)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)

        logits = self.out_conv(u1)
        return logits
