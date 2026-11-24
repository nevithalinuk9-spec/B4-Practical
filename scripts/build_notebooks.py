"""Generate the analysis notebooks (one per hypothesis).

    poetry run python scripts/build_notebooks.py notebooks
    poetry run python scripts/make_figures.py
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])

SETUP = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

from dip.analysis import (IMAGES, INK, MODEL_COLORS, MODEL_LABELS, MUTED, PALETTE,
                          load_runs, noisy_baseline, savefig, setup_style, summarize)
from dip.utils import DATA_DIR, RESULTS_DIR, add_gaussian_noise, load_image, to_numpy_image

setup_style()
pd.set_option("display.precision", 2)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.cells = out
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


# ---------------------------------------------------------------- H1
h1 = nb([
    ("md", """# H1 — Does PSNR rise, peak, then fall?

**Hypothesis.** While DIP is fitted to the noisy image, PSNR against the clean image rises, peaks and then falls. The drop after the peak is larger at higher noise.

**Data.** `results/h1/`: U-Net, 4 Kodak images × σ ∈ {25, 50}, seed 0, plus seeds 1–2 on `kodim03`, σ = 25. 10 % of pixels are held out from the loss (needed for H4)."""),
    ("code", SETUP),
    ("code", """h1 = load_runs("h1")
summary = summarize(h1, ["image", "sigma", "seed"])
base = pd.DataFrame([noisy_baseline(im, s) for im in IMAGES for s in (25, 50)])
summary = summary.merge(base, on=["image", "sigma"])
summary["gain_psnr"] = summary["peak_psnr"] - summary["noisy_psnr"]
summary[["image", "sigma", "seed", "noisy_psnr", "peak_psnr", "peak_iter", "final_psnr", "drop_psnr",
         "gain_psnr", "noisy_ssim", "peak_ssim", "peak_iter_ssim", "final_ssim"]]"""),
    ("md", "**Drop after the peak, σ = 25 vs. σ = 50** (seed 0):"),
    ("code", """seed0 = summary[summary.seed == 0]
drop = seed0.pivot(index="image", columns="sigma", values="drop_psnr")
drop.loc["mean"] = drop.mean()
drop"""),
    ("md", "**Run-to-run variability** (3 seeds, `kodim03`, σ = 25): differences between settings smaller than this spread should not be interpreted."),
    ("code", """seeds = summary[(summary.image == "kodim03") & (summary.sigma == 25)]
seeds[["seed", "peak_psnr", "peak_iter", "final_psnr"]].agg(["mean", "std", "min", "max"])"""),
    ("code", """fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
for col, sigma in enumerate((25, 50)):
    for k, im in enumerate(IMAGES):
        g = h1[(h1.image == im) & (h1.sigma == sigma) & (h1.seed == 0)]
        for row, metric in enumerate(("psnr_clean", "ssim_clean")):
            ax = axes[row, col]
            ax.plot(g["iter"], g[metric], color=PALETTE[k], label=im)
            i = g[metric].idxmax()
            ax.plot(g.loc[i, "iter"], g.loc[i, metric], "o", ms=5, color=PALETTE[k], mec="white", mew=1.2)
    axes[0, col].set_title(f"σ = {sigma}")
axes[0, 0].set_ylabel("PSNR vs. clean (dB)")
axes[1, 0].set_ylabel("SSIM vs. clean")
for ax in axes[1]:
    ax.set_xlabel("iteration")
axes[0, 1].legend(loc="upper right", ncols=2)
fig.suptitle("H1: quality against the clean image while fitting the noisy image (dots = peak)", x=0.02, ha="left")
fig.tight_layout()
savefig(fig, "h1_curves")"""),
    ("md", """**Teaser figure.** Clean, noisy, DIP output near the peak and at the end, for `kodim03`, σ = 25. Output snapshots exist only at iterations 100, 500, 1000, 2000 and 3000. We use the one closest to the PSNR peak."""),
    ("code", """run = "unet_kodim03_noisy_s25_seed0"
g = h1[h1.run == run]
peak_iter = int(g.loc[g["psnr_clean"].idxmax(), "iter"])
snap_iters = np.array([100, 500, 1000, 2000, 3000])
near = int(snap_iters[np.abs(snap_iters - peak_iter).argmin()])

x = load_image(DATA_DIR / "kodim03.png")
y = add_gaussian_noise(x, 25, seed=0)
def psnr_at(it):
    return g.loc[(g["iter"] - it).abs().idxmin(), "psnr_clean"]
panels = [
    ("clean", to_numpy_image(x), None),
    ("noisy (σ = 25)", to_numpy_image(y), seed0[(seed0.image == "kodim03") & (seed0.sigma == 25)].noisy_psnr.item()),
    (f"DIP, iteration {near}", io.imread(RESULTS_DIR / "h1" / "img" / f"{run}_it{near}.png") / 255, psnr_at(near)),
    ("DIP, iteration 3000", io.imread(RESULTS_DIR / "h1" / "img" / f"{run}_it3000.png") / 255, psnr_at(3000)),
]
r0, c0, s = 120, 30, 64  # zoom window (row, col, size)
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), gridspec_kw={"height_ratios": [1, 1]})
for k, (title, img, p) in enumerate(panels):
    axes[0, k].imshow(img)
    axes[0, k].add_patch(plt.Rectangle((c0, r0), s, s, fill=False, ec="#eb6834", lw=1.2))
    axes[0, k].set_title(title if p is None else f"{title}\\n{p:.1f} dB")
    axes[1, k].imshow(img[r0:r0 + s, c0:c0 + s], interpolation="nearest")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.tight_layout()
savefig(fig, "teaser")"""),
    ("md", """## Findings

- **H1 is confirmed.** Every run shows a clear PSNR peak followed by a decline.
- **The drop grows with noise:** on average about 1.6 dB at σ = 25 vs. 4.5 dB at σ = 50, and the peak comes much earlier at σ = 50.
- **Image content matters.** For the textured images `kodim05` and `kodim13`, the peak at σ = 25 comes late (~1900–2000 iterations) and the drop within 3000 iterations is small. Fine texture behaves like high-frequency content and is fitted late, together with the noise.
- **Seed variability is ~0.1 dB**, so differences of 0.5 dB or more are meaningful.
- **SSIM mostly agrees with PSNR,** with one caveat: SSIM shows a short spike after 20–40 iterations, when the output is still a very smooth image. On images with large smooth regions, SSIM briefly rewards this blur even though PSNR is low. The metrics are reported side by side for this reason."""),
])

# ---------------------------------------------------------------- H2
h2 = nb([
    ("md", """# H2 — Is a natural image fitted faster than noise?

**Hypothesis.** The same untrained U-Net fits a natural image much faster than the same pixels shuffled (same colours, no structure) or uniform noise.

**Measurement.** Training MSE against the target. Fitting speed = iterations until MSE < 0.01, plus MSE at iteration 500. No noise is added and no pixels are held out."""),
    ("code", SETUP),
    ("code", """h2 = load_runs("h2")
TARGET_LABEL = {"clean": "natural image", "shuffled": "shuffled pixels", "noise": "uniform noise"}
TARGET_COLOR = {"clean": PALETTE[0], "shuffled": PALETTE[1], "noise": PALETTE[2]}
THRESH = 0.01

def speed(g):
    hit = g.loc[g["loss"] < THRESH, "iter"]
    return pd.Series({
        "iters_to_mse<0.01": int(hit.iloc[0]) if len(hit) else np.nan,
        "mse@500": g.loc[g["iter"] == 500, "loss"].item(),
        "final_mse": g["loss"].iloc[-1],
        "final_psnr_vs_target": g["psnr_clean"].iloc[-1],
    })
table = h2.groupby(["image", "target"]).apply(speed, include_groups=False).reset_index()
table  # NaN = threshold never reached within 3000 iterations"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, im in zip(axes, ["kodim03", "kodim13"]):
    for t in ["clean", "shuffled", "noise"]:
        g = h2[(h2.image == im) & (h2.target == t)]
        ax.plot(g["iter"], g["loss"], color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    ax.axhline(THRESH, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_title(im)
    ax.set_xlabel("iteration")
axes[0].set_ylabel("training MSE vs. target (log)")
axes[0].text(3000, THRESH * 0.88, "MSE = 0.01", color=MUTED, ha="right", va="top", fontsize=8)
axes[1].legend(loc="lower right")
fig.suptitle("H2: the same network fits structure far faster than noise", x=0.02, ha="left")
fig.tight_layout()
savefig(fig, "h2_fitting_speed")"""),
    ("md", """## Findings

- **H2 is confirmed** (this reproduces Fig. 2 of Ulyanov et al., 2018).
- The natural image drops below MSE 0.01 after 20 (`kodim03`) and 130 (`kodim13`) iterations. Shuffled pixels reach it only after 2700 iterations (`kodim13`) or not at all (`kodim03`). Uniform noise never reaches it within 3000 iterations.
- After 3000 iterations, the fit is 36.5 dB (`kodim03`) / 28.1 dB (`kodim13`) for the natural images, compared with about 19–21 dB for shuffled pixels and about 15 dB for noise.
- **Shuffling destroys the speed-up** even though the colours are identical. So the speed comes from spatial structure, not from colour statistics.
- The textured `kodim13` is fitted more slowly than `kodim03`, consistent with H1: fine texture is "late" content."""),
])

# ---------------------------------------------------------------- H3
h3 = nb([
    ("md", """# H3 — Which part of the convolutional architecture is responsible?

| Model | Kernels | Upsampling | Tests |
|---|---|---|---|
| `unet` | 3×3 | bilinear | baseline |
| `unet_1x1` | 1×1 | bilinear | H3a: are spatial kernels needed? |
| `unet_1x1_wide` | 1×1, 170 channels (same #params as baseline) | bilinear | H3a, controls for parameter count |
| `unet_nearest` | 3×3 | nearest-neighbour | H3b: is smooth upsampling needed? |
| `mlp` | — | — | H3c: no spatial structure at all |

σ = 25, 4 images, seed 0. Each model uses its best learning rate from the sweep. **No held-out pixels:** models without spatial mixing cannot fill them in, which would penalise them twice."""),
    ("code", SETUP),
    ("md", "**Learning-rate sweep** (`kodim03`, peak PSNR in dB). The best value per row is used in H3."),
    ("code", """sweep = load_runs("lr_sweep")
sw = summarize(sweep, ["model", "lr"]).pivot(index="model", columns="lr", values="peak_psnr")
sw.loc[list(MODEL_LABELS)]"""),
    ("code", """h3 = load_runs("h3")
s3 = summarize(h3, ["model", "image"])
base = pd.DataFrame([noisy_baseline(im, 25) for im in IMAGES])
s3 = s3.merge(base, on="image")
params = h3.groupby("model")["params"].first()
lrs = h3.groupby("model")["lr"].first()
order = list(MODEL_LABELS)
agg = s3.groupby("model")[["peak_psnr", "drop_psnr", "peak_iter", "peak_ssim"]].mean().loc[order]
agg.insert(0, "lr", lrs.loc[order])
agg.insert(0, "params", params.loc[order])
agg["peak_vs_noisy"] = (s3.assign(d=s3.peak_psnr - s3.noisy_psnr).groupby("model")["d"].mean()).loc[order]
agg  # means over the 4 images"""),
    ("code", """s3.pivot(index="model", columns="image", values="peak_psnr").loc[order]"""),
    ("code", """fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
for ax, im in zip(axes.ravel(), IMAGES):
    for m in order:
        g = h3[(h3.model == m) & (h3.image == im)]
        ax.plot(g["iter"], g["psnr_clean"], color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    nb_ = base.loc[base.image == im, "noisy_psnr"].item()
    ax.axhline(nb_, color=MUTED, lw=0.8, ls="--")
    ax.set_title(im)
axes[0, 0].text(2950, base.loc[base.image == "kodim03", "noisy_psnr"].item() - 0.5, "noisy input", color=MUTED, fontsize=8, ha="right", va="top")
for ax in axes[:, 0]:
    ax.set_ylabel("PSNR vs. clean (dB)")
for ax in axes[1]:
    ax.set_xlabel("iteration")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
savefig(fig, "h3_curves")"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
xs = np.arange(len(order))
for ax, col, title in [(axes[0], "peak_psnr", "Peak PSNR (dB)"), (axes[1], "drop_psnr", "Drop after the peak (dB)")]:
    for i, m in enumerate(order):
        vals = s3.loc[s3.model == m, col].to_numpy()
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=22,
                   color=MODEL_COLORS[m], edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
    ax.set_xticks(xs, ["U-Net", "1×1", "1×1 wide", "nearest", "MLP"])
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
axes[0].axhline(base["noisy_psnr"].mean(), color=MUTED, lw=0.8, ls="--")
axes[0].text(-0.45, base["noisy_psnr"].mean() + 0.25, "noisy input", color=MUTED, fontsize=8, ha="left", va="bottom")
fig.suptitle("H3: dots = images, bar = mean over 4 images", x=0.02, ha="left")
fig.tight_layout()
savefig(fig, "h3_summary")"""),
    ("md", "**Visual comparison** (`kodim03`, zoomed crop). Each model is shown at the stored snapshot closest to its own PSNR peak."),
    ("code", """snap_iters = np.array([100, 500, 1000, 2000, 3000])
x = load_image(DATA_DIR / "kodim03.png")
y = add_gaussian_noise(x, 25, seed=0)
r0, c0, s = 120, 30, 64
tiles = [("clean", to_numpy_image(x)), ("noisy", to_numpy_image(y))]
for m in order:
    g = h3[(h3.model == m) & (h3.image == "kodim03")]
    run = g["run"].iloc[0]
    it = int(snap_iters[np.abs(snap_iters - g.loc[g["psnr_clean"].idxmax(), "iter"]).argmin()])
    img = io.imread(RESULTS_DIR / "h3" / "img" / f"{run}_it{it}.png") / 255
    p = g.loc[(g["iter"] - it).abs().idxmin(), "psnr_clean"]
    short = {"unet": "U-Net", "unet_1x1": "1×1", "unet_1x1_wide": "1×1 wide", "unet_nearest": "nearest", "mlp": "MLP"}[m]
    tiles.append((f"{short}\\nit {it}, {p:.1f} dB", img))
fig, axes = plt.subplots(1, len(tiles), figsize=(7.2, 1.9))
for ax, (title, img) in zip(axes, tiles):
    ax.imshow(img[r0:r0 + s, c0:c0 + s], interpolation="nearest")
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.tight_layout()
savefig(fig, "h3_crops")"""),
    ("md", """## Findings

**H3a — spatial kernels: confirmed.**
- With only 1×1 kernels, the mean peak falls by about 6 dB (69k params). With the parameter-matched wide variant (454k params) it still falls by about 4 dB.
- The 1×1 network does not beat the noisy input on average (−0.4 dB), and the wide one only by 1.6 dB. The U-Net gains about 6 dB.
- So the loss is not explained by parameter count.
- This is *against* the prediction from the literature (Heckel & Hand: 1×1 + upsampling suffices). One likely reason: the Deep Decoder upsamples after every layer and has no skip connections. Our 1×1 U-Net can route per-pixel information through the 1×1 skip path, which has no spatial prior.

**H3b — smooth upsampling: refuted for the peak, but it controls the decline.**
- With nearest-neighbour upsampling, the peak is unchanged (≈ 26.6 vs. 26.4 dB).
- But it reaches the peak about 3× earlier (≈ 490 vs. 1620 iterations) and then collapses about 4× harder (≈ 5.2 dB vs. 1.4 dB drop), ending near 21 dB on every image.
- So the 3×3 kernels determine *how good* the best output is. Bilinear upsampling determines *how slowly* noise is fitted afterwards, i.e. how forgiving the stopping point is.

**H3c — no spatial structure: confirmed.**
- The MLP never gets meaningfully above the noisy input (~20.7 dB on every image, +0.3 dB on average).
- It reproduces the noisy image almost immediately: training MSE ≈ 0.0002 by iteration 500, far below σ² ≈ 0.0096.

**Caveat.** The 1×1 variants are still *rising* at iteration 3000 (see the curves), so "weaker prior" and "slower optimisation" are not separated by these runs alone. The long runs at the end of this notebook address this."""),
])

h3.cells += [
    nbf.v4.new_markdown_cell("""## Long runs: weaker prior or just slower?

The 1×1 variants were still improving at iteration 3000. `h3_long` runs both of them for 10,000 iterations on `kodim03` (σ = 25), with the same learning rates. If they eventually match the U-Net, the difference is about optimisation speed. If they stay clearly below it, the difference is about the prior."""),
    nbf.v4.new_code_cell("""long = load_runs("h3_long")
ref = h3[(h3.model == "unet") & (h3.image == "kodim03")]
nb03 = noisy_baseline("kodim03", 25)["noisy_psnr"]
rows = []
for m, g in [("unet (3000 it.)", ref)] + list(long.groupby("model")):
    i = g["psnr_clean"].idxmax()
    rows.append({"model": m, "peak_psnr": g.loc[i, "psnr_clean"], "peak_iter": int(g.loc[i, "iter"]),
                 "psnr@3000": g.loc[(g["iter"] - 3000).abs().idxmin(), "psnr_clean"],
                 "final_psnr": g["psnr_clean"].iloc[-1], "final_train_mse": g["loss"].iloc[-1]})
print(f"noisy input: {nb03:.2f} dB, sigma^2 = {(25/255)**2:.4f}")
pd.DataFrame(rows)"""),
    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.plot(ref["iter"], ref["psnr_clean"], color=MODEL_COLORS["unet"], label=MODEL_LABELS["unet"] + " (3000 it.)")
for m, g in long.groupby("model"):
    sm = g["psnr_clean"].rolling(5, center=True, min_periods=1).mean()
    ax.plot(g["iter"], sm, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
ax.axhline(nb03, color=MUTED, lw=0.8, ls="--")
ax.text(9900, nb03 - 0.4, "noisy input", color=MUTED, fontsize=8, ha="right", va="top")
ax.set_xlabel("iteration")
ax.set_ylabel("PSNR vs. clean (dB)")
ax.legend(loc="lower right")
ax.set_title("kodim03, σ = 25: 1×1 variants run 3× longer (moving average of 5 log points)", loc="left")
fig.tight_layout()
savefig(fig, "h3_long")"""),
    nbf.v4.new_markdown_cell("""## Findings (long runs)

- **Both 1×1 variants are still improving after 10,000 iterations.** The parameter-matched one reaches 27.5 dB, halving its gap to the U-Net (29.1 dB) from about 3.3 dB at iteration 3000 to 1.6 dB. The small one reaches 25.6 dB.
- **The U-Net's training MSE falls below σ² after about 630 iterations; the wide 1×1 variant's never does within 10,000** (it ends at 0.0098 vs. σ² = 0.0096). Without spatial kernels, the image content is fitted more than 15× more slowly.
- **H3a therefore has to be refined.** Spatial kernels are clearly necessary for DIP to work within a normal iteration budget, mostly because they make structure fit *fast*. Whether a 1×1 network would eventually reach the same quality is open: the remaining 1.6 dB gap may or may not close with even longer training.
- *Note:* these two runs were made on Apple MPS, not CUDA. At iteration 3000 the wide variant scores 25.5 dB here vs. 25.8 dB in the CUDA run, i.e. about 0.3 dB of hardware non-determinism."""),
]

# ---------------------------------------------------------------- H4 + H5
h4 = nb([
    ("md", """# H4 — Can we choose when to stop without the clean image? (+ H5: EMA)

Rules (see `src/dip/stopping.py`), applied to the 10 `h1` runs:

| Rule | Uses | Idea |
|---|---|---|
| oracle | clean image | best PSNR (upper bound) |
| fixed (1500) | — | always stop at iteration 1500 |
| noise level (true σ) | σ | first iteration where smoothed training MSE < σ² (discrepancy principle) |
| noise level (estimated σ) | noisy image only | same, with σ estimated by `skimage.restoration.estimate_sigma` |
| held-out pixels | — | minimum of the smoothed loss on the 10 % held-out pixels |

Reported: **PSNR (and SSIM) lost compared with the oracle**."""),
    ("code", SETUP + """
from dip.stopping import estimate_noise_sigma, evaluate_rules"""),
    ("code", """h1 = load_runs("h1")
rows = []
for run, g in h1.groupby("run"):
    g = g.reset_index(drop=True)
    im, sigma, seed = g.loc[0, "image"], int(g.loc[0, "sigma"]), int(g.loc[0, "seed"])
    y = add_gaussian_noise(load_image(DATA_DIR / f"{im}.png"), sigma, seed=seed)
    est = estimate_noise_sigma(y)
    r = evaluate_rules(g, sigma / 255, est)
    r["image"], r["sigma"], r["seed"], r["sigma_est"] = im, sigma, seed, est * 255
    rows.append(r)
R = pd.concat(rows, ignore_index=True)
RULES = ["fixed (1500)", "noise level (true σ)", "noise level (estimated σ)", "held-out pixels"]
R.groupby(["image", "sigma", "seed"])["sigma_est"].first()  # σ estimate vs. true σ"""),
    ("code", """R[R.rule != "oracle"].groupby("rule")[["psnr_lost", "ssim_lost"]].agg(["mean", "median", "max"]).loc[RULES]"""),
    ("code", """R.pivot_table(index=["image", "sigma", "seed"], columns="rule", values="iter").loc[:, ["oracle"] + RULES]  # chosen iteration"""),
    ("code", """RULE_COLORS = dict(zip(RULES, PALETTE[1:5]))
SHORT = {"fixed (1500)": "fixed 1500", "noise level (true σ)": "noise level\\n(true σ)",
         "noise level (estimated σ)": "noise level\\n(estimated σ)", "held-out pixels": "held-out\\npixels"}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for i, rule in enumerate(RULES):
    v = R.loc[R.rule == rule, "psnr_lost"].to_numpy()
    jitter = np.linspace(-0.15, 0.15, len(v))
    ax.scatter(i + jitter, v, s=20, color=RULE_COLORS[rule], edgecolor="white", linewidth=0.8, zorder=3)
    ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.6, zorder=4)
ax.set_xticks(range(len(RULES)), [SHORT[r] for r in RULES])
ax.set_ylabel("PSNR lost vs. oracle (dB)")
ax.grid(axis="x", visible=False)
ax.set_title("H4: 10 runs per rule (bar = median)", loc="left")
fig.tight_layout()
savefig(fig, "h4_psnr_lost")"""),
    ("md", """**Example run** (`kodim03`, σ = 50) showing the signals each rule uses.
- **Left:** PSNR with the chosen stopping points.
- **Middle:** training MSE and the σ² line.
- **Right:** held-out loss vs. the true error on the whole image plus σ²."""),
    ("code", """g = h1[(h1.image == "kodim03") & (h1.sigma == 50) & (h1.seed == 0)].reset_index(drop=True)
r = R[(R.image == "kodim03") & (R.sigma == 50) & (R.seed == 0)].set_index("rule")
s2 = (50 / 255) ** 2
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
ax = axes[0]
ax.plot(g["iter"], g["psnr_clean"], color=PALETTE[0])
for rule in ["oracle"] + RULES:
    it = r.loc[rule, "iter"]; p = r.loc[rule, "psnr"]
    c = INK if rule == "oracle" else RULE_COLORS[rule]
    ax.plot(it, p, "o", ms=6, color=c, mec="white", mew=1.2, label=rule, zorder=4)
ax.set_title("PSNR vs. clean (dB)", loc="left")
handles, labels = ax.get_legend_handles_labels()
ax = axes[1]
ax.plot(g["iter"], g["loss"], color=PALETTE[0])
ax.axhline(s2, color=MUTED, ls="--", lw=0.8)
ax.text(3000, s2 * 1.04, "σ²", color=MUTED, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_title("training MSE", loc="left")
ax = axes[2]
true_mse = 10 ** (-g["psnr_clean"] / 10)
ax.plot(g["iter"], g["mse_holdout"].rolling(10, center=True, min_periods=1).mean(), color=RULE_COLORS["held-out pixels"], label="held-out loss")
ax.plot(g["iter"], true_mse + s2, color=INK, lw=1.2, ls="--", label="true MSE + σ²")
ax.set_title("held-out loss", loc="left")
ax.legend(fontsize=7)
for a in axes:
    a.set_xlabel("iteration")
fig.tight_layout()
fig.legend(handles, labels, loc="upper center", ncols=5, fontsize=7.5, bbox_to_anchor=(0.5, 0.0), title="stopping point chosen by", title_fontsize=7.5)
savefig(fig, "h4_example")"""),
    ("md", """## Findings (H4)

**Noise-level rule (discrepancy principle): works very well.**
- Median loss of about 0.2 dB vs. the oracle, at most about 0.6 dB.
- Estimating σ from the noisy image is just as good: the estimates are within about 2 units of the true σ (on the 0–255 scale). In our setting the rule needs no extra information at all.

**Held-out pixels: works, but systematically stops too late.**
- About 1 dB lost on average, up to 2 dB.
- Reason: the identity *held-out loss = error on held-out pixels + σ²* holds, but the error *on the held-out pixels* is not the error on the whole image. The held-out pixels are never fitted directly; they are only inpainted from their neighbours. So noise reaches them later, and their error keeps falling after the full-image error has started to rise (right panel).

**Fixed budget: worst.**
- About 1.6 dB lost on average, up to 4.4 dB.
- The best iteration ranges from about 340 to 2000 depending on image and σ, so no single budget fits all.

**Answer to the second half of the research question:** yes. A stopping rule that uses only the noisy image (noise level with estimated σ) gets within about 0.25 dB of the oracle on these images."""),
    ("md", "## H5 — Does averaging outputs (EMA) make the stopping point matter less?"),
    ("code", """s_plain = summarize(h1, ["image", "sigma", "seed"])
ema = h1.assign(psnr_clean=h1["psnr_ema"], ssim_clean=h1["ssim_ema"])
s_ema = summarize(ema, ["image", "sigma", "seed"])
h5 = s_plain[["image", "sigma", "seed", "peak_psnr", "drop_psnr"]].merge(
    s_ema[["image", "sigma", "seed", "peak_psnr", "drop_psnr"]], on=["image", "sigma", "seed"], suffixes=("", "_ema"))
h5.loc["mean"] = h5[["peak_psnr", "drop_psnr", "peak_psnr_ema", "drop_psnr_ema"]].mean()
h5"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
for ax, sigma in zip(axes, (25, 50)):
    g = h1[(h1.image == "kodim03") & (h1.sigma == sigma) & (h1.seed == 0)]
    ax.plot(g["iter"], g["psnr_clean"], color=PALETTE[0], label="plain output")
    ax.plot(g["iter"], g["psnr_ema"], color=PALETTE[1], label="EMA of outputs (0.99)")
    ax.set_title(f"kodim03, σ = {sigma}")
    ax.set_xlabel("iteration")
axes[0].set_ylabel("PSNR vs. clean (dB)")
axes[1].legend(loc="lower left")
fig.suptitle("H5: averaging lifts the whole curve but does not flatten the decline", x=0.02, ha="left")
fig.tight_layout()
savefig(fig, "h5_ema")"""),
    ("md", """## Findings (H5)

**H5 as stated is refuted, but averaging still helps a lot.**
- The drop after the peak is **not** smaller with EMA: 3.3 dB vs. 3.1 dB on average. So the stopping point matters just as much.
- But the EMA output is about **2 dB better at every point of the curve**: mean peak 27.2 vs. 25.1 dB, and the final value is also about 1.9 dB higher.
- A likely reason: every plain output is computed from a randomly perturbed input `z` (std 1/30), so it carries extra step-to-step jitter. Averaging about 100 consecutive outputs cancels that jitter, much like an ensemble.
- Practical consequence: stopping rules should be applied to the EMA output. Combined with the noise-level rule, this would give results about 2 dB above the plain oracle (not evaluated separately here).
- This is a simplification *inspired by* Cheng et al. (2019), who average SGLD posterior samples. It is not a reproduction of their method."""),
])

for name, n in [("01_H1_peak", h1), ("02_H2_fitting_speed", h2), ("03_H3_architectures", h3), ("04_H4_H5_stopping", h4)]:
    nbf.write(n, OUT / f"{name}.ipynb")
    print("wrote", name)
