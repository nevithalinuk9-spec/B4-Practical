"""Small helpers: seeding, device, image IO, noise, PSNR."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from skimage import io, transform
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def center_crop_resize(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Crop the largest centred square, then resize to size×size. Returns float in [0, 1]."""
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    img = img[top : top + s, left : left + s]
    return transform.resize(img, (size, size), anti_aliasing=True).astype(np.float32)


def load_image(path: str | Path) -> torch.Tensor:
    """Load an RGB image in [0, 1] as a tensor of shape (1, 3, H, W)."""
    img = io.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    img = img[..., :3]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor → (H, W, 3) numpy array clipped to [0, 1] (for plotting)."""
    return x.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def add_gaussian_noise(x: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    """Add N(0, (sigma/255)^2) noise. sigma is on the 0–255 scale.

    The result is NOT clipped to [0, 1], so the noise stays exactly Gaussian
    (this matters for the noise-level stopping rule, which assumes variance sigma^2).
    """
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g) * (float(sigma) / 255.0)
    return x + noise


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB for images in [0, 1]. `a` is clipped to [0, 1] first."""
    mse = torch.mean((a.clamp(0, 1) - b) ** 2).item()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM (Wang et al. 2004) for images in [0, 1]. `a` is clipped to [0, 1] first."""
    return float(structural_similarity(to_numpy_image(a), to_numpy_image(b), channel_axis=2, data_range=1.0))


def shuffle_pixels(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Randomly permute pixel positions (keeps colour statistics, destroys structure)."""
    _, c, h, w = x.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(h * w, generator=g)
    return x.reshape(1, c, h * w)[:, :, perm].reshape(1, c, h, w)
