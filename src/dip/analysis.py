"""Shared helpers for the analysis notebooks: loading logs, summary tables, figure style."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from .utils import DATA_DIR, FIGURES_DIR, RESULTS_DIR, add_gaussian_noise, load_image, psnr, ssim

# Categorical colours in fixed order (colour-blind-checked palette). Each model keeps its colour
# in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = dict(zip(["unet", "unet_1x1", "unet_1x1_wide", "unet_nearest", "mlp"], PALETTE))
MODEL_LABELS = {
    "unet": "U-Net (3×3, bilinear)",
    "unet_1x1": "U-Net 1×1",
    "unet_1x1_wide": "U-Net 1×1, wide",
    "unet_nearest": "U-Net, nearest upsampling",
    "mlp": "MLP",
}
IMAGES = ["kodim03", "kodim05", "kodim13", "kodim15"]
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3de"


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def savefig(fig: plt.Figure, name: str) -> None:
    """Save as PDF (for the Quarto note) and PNG (for slides) into figures/."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")


def load_runs(exp: str) -> pd.DataFrame:
    """All CSV logs of one experiment in one long table, with a `run` column (file name)."""
    files = sorted((RESULTS_DIR / exp).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no results in {RESULTS_DIR / exp}; run scripts/run_all.py --exp {exp}")
    return pd.concat([pd.read_csv(f).assign(run=f.stem) for f in files], ignore_index=True)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per run: peak PSNR and its iteration, final PSNR, drop after the peak, same for SSIM."""
    def one(g: pd.DataFrame) -> pd.Series:
        i, j = g["psnr_clean"].idxmax(), g["ssim_clean"].idxmax()
        return pd.Series({
            "peak_psnr": g.loc[i, "psnr_clean"],
            "peak_iter": int(g.loc[i, "iter"]),
            "final_psnr": g["psnr_clean"].iloc[-1],
            "drop_psnr": g.loc[i, "psnr_clean"] - g["psnr_clean"].iloc[-1],
            "peak_ssim": g.loc[j, "ssim_clean"],
            "peak_iter_ssim": int(g.loc[j, "iter"]),
            "final_ssim": g["ssim_clean"].iloc[-1],
        })
    return df.groupby(by).apply(one, include_groups=False).reset_index()


def noisy_baseline(image: str, sigma: float, seed: int = 0) -> dict:
    """PSNR/SSIM of the noisy input itself: the 'do nothing' reference."""
    x = load_image(DATA_DIR / f"{image}.png")
    y = add_gaussian_noise(x, sigma, seed=seed)
    return {"image": image, "sigma": sigma, "noisy_psnr": psnr(y, x), "noisy_ssim": ssim(y, x)}
