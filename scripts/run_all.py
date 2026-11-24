"""Run the experiments and write one CSV log per run to results/<exp>/.

Examples:
    poetry run python scripts/run_all.py --exp h1
    poetry run python scripts/run_all.py --exp h3 --iters 3000
    poetry run python scripts/run_all.py --exp all --quick

Runs whose CSV already exists are skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from skimage import io

from dip.fit import fit
from dip.models import build_model, count_params
from dip.utils import DATA_DIR, RESULTS_DIR, add_gaussian_noise, load_image, shuffle_pixels, to_numpy_image

IMAGES = ["kodim03", "kodim05", "kodim13", "kodim15"]
SNAPSHOTS = (100, 500, 1000, 2000, 3000)

# Learning rate per architecture. Update after the lr sweep (--exp lr_sweep).
LR = {"unet": 0.01, "unet_1x1": 0.01, "unet_1x1_wide": 0.01, "unet_nearest": 0.01, "mlp": 0.001}
H3_MODELS = ("unet", "unet_1x1", "unet_1x1_wide", "unet_nearest", "mlp")


def experiments(name: str) -> list[dict]:
    """Return the list of runs for one experiment. Each run is a dict of settings."""
    runs: list[dict] = []
    if name == "h1":
        # PSNR peak: every image × two noise levels, plus 2 extra seeds on one setting
        for img in IMAGES:
            for sigma in (25, 50):
                runs.append(dict(model="unet", image=img, target="noisy", sigma=sigma, seed=0))
        for seed in (1, 2):
            runs.append(dict(model="unet", image="kodim03", target="noisy", sigma=25, seed=seed))
    elif name == "h2":
        # Natural image vs. shuffled pixels vs. pure noise (no added noise, no held-out pixels)
        for img in ("kodim03", "kodim13"):
            for target in ("clean", "shuffled", "noise"):
                runs.append(dict(model="unet", image=img, target=target, sigma=0, seed=0, holdout_frac=0.0))
    elif name == "h3":
        # Architecture ablation at sigma = 25 (kernels: H3a, upsampling: H3b, no structure: MLP)
        for model in H3_MODELS:
            for img in IMAGES:
                runs.append(dict(model=model, image=img, target="noisy", sigma=25, seed=0))
    elif name == "lr_sweep":
        # Fair comparison for H3: try a few learning rates per architecture on one image
        for model in H3_MODELS:
            for lr in (0.001, 0.003, 0.01):
                runs.append(dict(model=model, image="kodim03", target="noisy", sigma=25, seed=0, lr=lr))
    else:
        raise ValueError(f"unknown experiment: {name}")
    return runs


def run_name(r: dict) -> str:
    name = f"{r['model']}_{r['image']}_{r['target']}_s{r['sigma']}_seed{r['seed']}"
    if "lr" in r:
        name += f"_lr{r['lr']}"
    return name


def make_target(r: dict, x: torch.Tensor) -> torch.Tensor:
    if r["target"] == "noisy":
        return add_gaussian_noise(x, r["sigma"], seed=r["seed"])
    if r["target"] == "clean":
        return x
    if r["target"] == "shuffled":
        return shuffle_pixels(x, seed=r["seed"])
    if r["target"] == "noise":
        g = torch.Generator().manual_seed(r["seed"])
        return torch.rand(x.shape, generator=g)
    raise ValueError(r["target"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="h1", help="h1 | h2 | h3 | lr_sweep | all")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--quick", action="store_true", help="smoke test: 1 run, 100 iterations, separate folder")
    args = p.parse_args()

    exps = ["h1", "h2", "h3", "lr_sweep"] if args.exp == "all" else [args.exp]
    for exp in exps:
        runs = experiments(exp)
        n_iter = args.iters
        out_dir = RESULTS_DIR / exp
        if args.quick:
            runs, n_iter, out_dir = runs[:1], 100, RESULTS_DIR / "quick" / exp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "img").mkdir(exist_ok=True)

        for i, r in enumerate(runs, 1):
            name = run_name(r)
            csv_path = out_dir / f"{name}.csv"
            if csv_path.exists() and not args.quick:
                print(f"[{exp} {i}/{len(runs)}] skip {name} (exists)")
                continue

            x = load_image(DATA_DIR / f"{r['image']}.png")
            y = make_target(r, x)
            model = build_model(r["model"], *x.shape[-2:])
            lr = r.get("lr", LR[r["model"]])

            t0 = time.time()
            res = fit(
                model, y,
                x_clean=x if r["target"] == "noisy" else y,
                n_iter=n_iter, lr=lr,
                holdout_frac=r.get("holdout_frac", 0.1),
                snapshot_iters=SNAPSHOTS, seed=r["seed"],
            )
            log = res.log
            for k, v in {**r, "lr": lr, "params": count_params(model)}.items():
                log[k] = v
            log.to_csv(csv_path, index=False)

            for it, img in res.snapshots.items():
                io.imsave(out_dir / "img" / f"{name}_it{it}.png",
                          (to_numpy_image(img) * 255).round().astype(np.uint8), check_contrast=False)

            best = log["psnr_clean"].max()
            print(f"[{exp} {i}/{len(runs)}] {name}: best PSNR {best:.2f} dB, "
                  f"final {log['psnr_clean'].iloc[-1]:.2f} dB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
