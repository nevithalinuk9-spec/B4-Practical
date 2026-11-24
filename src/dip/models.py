"""Generator networks for the experiments.

- UNet(kernel_size=3): simplified version of the skip "hourglass" network from
  Ulyanov et al. (2018). This is the baseline.
- UNet(kernel_size=1): identical network but every convolution is 1×1, so no
  convolution mixes neighbouring pixels. Down/upsampling is the only spatial
  operation left. (H3a)
- UNet(kernel_size=1, channels=170): same, but widened so the parameter count
  matches the baseline (~454k). Rules out "it only has fewer parameters". (H3a)
- UNet(upsample="nearest"): 3×3 kernels kept, but upsampling copies pixels
  instead of interpolating, so it adds no smoothing. (H3b)
- MLP: fully connected network from a fixed latent vector to all pixels. No
  convolutional or spatial structure at all. (H3)

Every model has `make_input(height, width, seed)` returning the fixed input z.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn_act(c_in: int, c_out: int, k: int, stride: int = 1) -> nn.Sequential:
    pad = (k - 1) // 2
    layers: list[nn.Module] = []
    if pad > 0:
        layers.append(nn.ReflectionPad2d(pad))
    layers += [
        nn.Conv2d(c_in, c_out, k, stride=stride, bias=True),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        channels: int = 64,
        levels: int = 4,
        skip_channels: int = 4,
        kernel_size: int = 3,
        upsample: str = "bilinear",
    ) -> None:
        super().__init__()
        if upsample not in ("bilinear", "nearest"):
            raise ValueError(f"unknown upsample mode: {upsample}")
        self.in_channels = in_channels
        self.upsample = upsample
        k = kernel_size
        self.down = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(levels):
            c_in = in_channels if i == 0 else channels
            # downsample with a strided conv, then one more conv
            self.down.append(nn.Sequential(conv_bn_act(c_in, channels, k, stride=2), conv_bn_act(channels, channels, k)))
            # skip connection taken before downsampling
            self.skip.append(conv_bn_act(c_in, skip_channels, 1))
            # after upsampling: concatenate skip, then conv k×k and 1×1
            self.up.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels + skip_channels),
                    conv_bn_act(channels + skip_channels, channels, k),
                    conv_bn_act(channels, channels, 1),
                )
            )
        self.out = nn.Conv2d(channels, out_channels, 1)

    def make_input(self, height: int, width: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.rand((1, self.in_channels, height, width), generator=g) * 0.1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skips = []
        h = z
        for down, skip in zip(self.down, self.skip):
            skips.append(skip(h))
            h = down(h)
        for i in reversed(range(len(self.up))):
            if self.upsample == "bilinear":
                h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            else:
                h = F.interpolate(h, size=skips[i].shape[-2:], mode="nearest")
            h = self.up[i](torch.cat([skips[i], h], dim=1))
        return torch.sigmoid(self.out(h))


class MLP(nn.Module):
    """Fixed latent vector → hidden layers → every pixel value (no spatial structure)."""

    def __init__(self, height: int = 256, width: int = 256, out_channels: int = 3,
                 latent_dim: int = 128, hidden: int = 256, depth: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = (out_channels, height, width)
        layers: list[nn.Module] = []
        d = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2, inplace=True)]
            d = hidden
        layers.append(nn.Linear(d, out_channels * height * width))
        self.net = nn.Sequential(*layers)

    def make_input(self, height: int, width: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.rand((1, self.latent_dim), generator=g) * 0.1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z)).view(1, *self.shape)


def build_model(name: str, height: int = 256, width: int = 256) -> nn.Module:
    if name == "unet":
        return UNet(kernel_size=3)
    if name == "unet_1x1":
        return UNet(kernel_size=1)
    if name == "unet_1x1_wide":
        return UNet(kernel_size=1, channels=170)  # ≈ same parameter count as "unet"
    if name == "unet_nearest":
        return UNet(upsample="nearest")
    if name == "mlp":
        return MLP(height=height, width=width)
    raise ValueError(f"unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
