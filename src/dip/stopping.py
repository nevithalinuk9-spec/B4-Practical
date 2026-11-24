"""Stopping rules for H4, applied after the fact to a logged run (one row per logged iteration).

Every rule returns the *row index* of the chosen stopping point in the log.
Only `stop_oracle` uses the clean image; all others use quantities that are
available in practice (training loss, held-out loss, and optionally σ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from skimage.restoration import estimate_sigma


def _trailing_mean(s: pd.Series, window: int) -> pd.Series:
    """Causal moving average: only uses past values (usable while training)."""
    return s.rolling(window, min_periods=1).mean()


def stop_oracle(log: pd.DataFrame) -> int:
    """Iteration with the highest PSNR against the clean image. Upper bound, not usable in practice."""
    return int(log["psnr_clean"].idxmax())


def stop_fixed(log: pd.DataFrame, iteration: int = 1500) -> int:
    """Always stop at the same iteration."""
    return int((log["iter"] - iteration).abs().idxmin())


def stop_discrepancy(log: pd.DataFrame, sigma: float, window: int = 5) -> int:
    """Morozov's discrepancy principle: stop the first time the (smoothed) training MSE drops below σ².

    `sigma` is on the 0–1 scale. If the loss never gets below σ², the last iteration is returned.
    """
    smoothed = _trailing_mean(log["loss"], window)
    below = np.flatnonzero(smoothed.to_numpy() < sigma**2)
    return int(log.index[below[0]]) if len(below) else int(log.index[-1])


def stop_holdout(log: pd.DataFrame, window: int = 10) -> int:
    """Self-validation: minimum of the (smoothed) loss on held-out pixels.

    The held-out loss equals the true error plus a constant (σ² per entry), so its minimum
    estimates the oracle stopping point. In practice this means: keep training a while past the
    minimum and keep the best output seen so far. A centred moving average removes the
    step-to-step jitter caused by the input perturbation.
    """
    smoothed = log["mse_holdout"].rolling(window, min_periods=1, center=True).mean()
    return int(smoothed.idxmin())


def estimate_noise_sigma(y: torch.Tensor) -> float:
    """Estimate Gaussian noise σ (0–1 scale) from the noisy image alone (wavelet-based, Donoho 1994)."""
    img = y.squeeze(0).permute(1, 2, 0).numpy()
    return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))


def evaluate_rules(log: pd.DataFrame, sigma_true: float, sigma_est: float | None = None,
                   fixed_iteration: int = 1500) -> pd.DataFrame:
    """Apply every rule to one run. Returns one row per rule with the chosen iteration and
    the PSNR/SSIM lost compared with the oracle. σ values are on the 0–1 scale."""
    rules = {
        "oracle": stop_oracle(log),
        f"fixed ({fixed_iteration})": stop_fixed(log, fixed_iteration),
        "noise level (true σ)": stop_discrepancy(log, sigma_true),
        "held-out pixels": stop_holdout(log),
    }
    if sigma_est is not None:
        rules["noise level (estimated σ)"] = stop_discrepancy(log, sigma_est)

    best = log.loc[rules["oracle"]]
    rows = []
    for name, idx in rules.items():
        r = log.loc[idx]
        rows.append({
            "rule": name,
            "iter": int(r["iter"]),
            "psnr": r["psnr_clean"],
            "ssim": r["ssim_clean"],
            "psnr_lost": best["psnr_clean"] - r["psnr_clean"],
            "ssim_lost": best["ssim_clean"] - r["ssim_clean"],
        })
    return pd.DataFrame(rows)
