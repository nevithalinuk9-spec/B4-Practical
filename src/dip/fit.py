"""The DIP optimisation loop.

Fits `model(z)` to a target image `y` with Adam and MSE, logging everything the
hypotheses need:

- loss          MSE vs. y on the training pixels (noise-level rule, H4)
- mse_holdout   MSE vs. y on held-out pixels never used in the loss (H4)
- psnr_clean    PSNR of the output vs. the clean image (H1, H3; oracle in H4)
- psnr_ema      same for an exponential moving average of outputs (H5)
- ssim_clean, ssim_ema   SSIM counterparts of the two PSNR columns
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import nn
from tqdm.auto import trange

from .utils import get_device, psnr, set_seed, ssim


@dataclass
class FitResult:
    log: pd.DataFrame
    final: torch.Tensor
    final_ema: torch.Tensor
    snapshots: dict[int, torch.Tensor] = field(default_factory=dict)


def fit(
    model: nn.Module,
    y: torch.Tensor,
    x_clean: torch.Tensor | None = None,
    n_iter: int = 3000,
    lr: float = 0.01,
    reg_noise_std: float = 1.0 / 30.0,
    holdout_frac: float = 0.1,
    ema_decay: float = 0.99,
    log_every: int = 10,
    snapshot_iters: tuple[int, ...] = (),
    seed: int = 0,
    device: torch.device | None = None,
    progress: bool = True,
) -> FitResult:
    device = device or get_device()
    set_seed(seed)
    _, _, h, w = y.shape

    model = model.to(device)
    z = model.make_input(h, w, seed=seed).to(device)
    y = y.to(device)
    x_clean = x_clean.to(device) if x_clean is not None else None

    # Held-out pixels: a random subset of pixel positions (all 3 channels) excluded from the loss.
    g = torch.Generator().manual_seed(seed + 12345)
    holdout = (torch.rand((1, 1, h, w), generator=g) < holdout_frac).to(device)
    train_mask = (~holdout).float()
    hold_mask = holdout.float()
    n_train = train_mask.sum() * y.shape[1]
    n_hold = max(hold_mask.sum().item() * y.shape[1], 1.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = None
    rows = []
    snapshots: dict[int, torch.Tensor] = {}

    for it in trange(1, n_iter + 1, disable=not progress, leave=False):
        z_in = z + reg_noise_std * torch.randn_like(z) if reg_noise_std > 0 else z
        out = model(z_in)
        sq = (out - y) ** 2
        loss = (sq * train_mask).sum() / n_train

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            out_d = out.detach()
            ema = out_d.clone() if ema is None else ema_decay * ema + (1 - ema_decay) * out_d
            if it % log_every == 0 or it == 1:
                row = {
                    "iter": it,
                    "loss": loss.item(),
                    "mse_holdout": ((sq.detach() * hold_mask).sum() / n_hold).item(),
                }
                if x_clean is not None:
                    row["psnr_clean"] = psnr(out_d, x_clean)
                    row["psnr_ema"] = psnr(ema, x_clean)
                    row["ssim_clean"] = ssim(out_d, x_clean)
                    row["ssim_ema"] = ssim(ema, x_clean)
                rows.append(row)
            if it in snapshot_iters:
                snapshots[it] = out_d.cpu()

    return FitResult(log=pd.DataFrame(rows), final=out_d.cpu(), final_ema=ema.cpu(), snapshots=snapshots)
