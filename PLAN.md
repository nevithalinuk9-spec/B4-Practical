# B4 · Architectural Priors in Image Restoration — Project Plan

> **Research question.** Is the convolutional architecture itself responsible for Deep Image Prior (DIP) fitting image structure before noise, and can we choose when to stop without access to the clean image?

**Scope:** a small, clean experiment that isolates the phenomenon. The course brief puts it this way: *"A small experiment that isolates a phenomenon clearly is often stronger than a large experiment that is hard to interpret."*

---

## 1. Deliverables

All three are assessed pass/fail. They are graded only if the live presentation is satisfactory.

| Artifact | Requirements |
|---|---|
| **Slide deck** | Coherent ~60-minute lecture: motivation, theory, literature, experiment, results, critical interpretation. Established results must be distinguished from our own findings. |
| **Reproducible repository** | README, environment definition (Poetry), data instructions, code + notebooks, scripts that regenerate every figure and table. |
| **Scientific note** | 3–4 pages excluding references, written in **Quarto**. Submit the `.qmd` source and the rendered PDF. It must stand on its own and must not be a transcript of the slides. |

**Key point about the method:** DIP needs **no training dataset**. A randomly initialised CNN is fitted to **one** noisy image. A few clean test images are needed only so that noise can be added synthetically and PSNR can be measured against the known ground truth.

---

## 2. Theoretical background

**Denoising as an inverse problem.** We observe `y = x + n` with `n ~ N(0, σ²I)` and want the clean image `x`. The classical approach adds a hand-designed regulariser `R` that encodes "what natural images look like" (e.g. total variation):

$$\hat{x} = \arg\min_x \; \lVert x - y \rVert^2 + \lambda R(x)$$

**Deep Image Prior.** DIP replaces `R` with a generator network `f_θ` and a fixed random input `z`, and optimises only the weights:

$$\theta^* = \arg\min_\theta \; \lVert f_\theta(z) - y \rVert^2, \qquad \hat{x} = f_{\theta^*}(z)$$

The network is over-parameterised and will eventually reproduce `y`, noise included. DIP therefore works only because of **implicit regularisation by early stopping**: the optimisation reaches natural-image content long before it reaches the noise.

**Why structure is fitted first (spectral bias).** Gradient descent on convolutional generators fits low-frequency components (smooth regions, edges, repeated patterns) much faster than high-frequency components. White noise spreads its energy over all frequencies, so most of it is fitted late. Heckel & Soltanolkotabi (2020) prove this for simplified convolutional generators. They identify the **upsampling operation** and the convolutional weight sharing as the source of the bias. Heckel & Hand (2019) show that a network with **only 1×1 convolutions plus upsampling** is already a strong image prior. The literature therefore predicts that the *upsampling* may matter more than the *size of the convolution kernels*. H3 is designed to test exactly this.

**Bayesian view.** Cheng et al. (2019) show that, in the infinite-width limit, a randomly initialised DIP network is a Gaussian process over images. Averaging over the posterior (via SGLD) avoids the need for early stopping. Our EMA experiment (H5) is a simple approximation *inspired by* this idea, not a reproduction of SGLD.

**Why held-out pixels give a valid stopping signal.** Let `M` be pixels excluded from the loss. The network never sees `n` on `M`, so its output there is independent of that noise. Then:

$$\mathbb{E}\big[\lVert f_\theta(z) - y \rVert_M^2\big] = \lVert f_\theta(z) - x \rVert_M^2 + |M|\,\sigma^2$$

The held-out loss equals the true error against the clean image plus a constant. It is therefore minimised at (approximately) the same iteration as the true error, without ever using `x`.

**Why the noise-level rule works.** A perfect reconstruction `f = x` has expected training loss `E‖x − y‖²/N = σ²`. A training MSE below `σ²` therefore means the network has started to reproduce noise. This is Morozov's discrepancy principle from classical inverse problems.

---

## 3. Hypotheses

| ID | Hypothesis | Measurement | Confirmed if… |
|---|---|---|---|
| **H1** | PSNR against the clean image rises, peaks, then falls | PSNR vs. iteration. Peak iteration `t*`, peak PSNR, drop `Δ = PSNR(t*) − PSNR(final)` | Clear peak; `Δ` larger at σ = 50 than at σ = 25 |
| **H2** | A natural image is fitted faster than shuffled pixels or pure noise | Iterations until training MSE < 0.01; training MSE at iteration 500 | The natural image reaches the threshold several times faster |
| **H3a** | Spatial convolution kernels are necessary for the effect | Peak PSNR and `Δ`: U-Net vs. 1×1 U-Net vs. parameter-matched 1×1 U-Net | 1×1 variants have a clearly lower peak (e.g. > 2 dB) |
| **H3b** | Smooth (bilinear) upsampling is necessary for the effect | Peak PSNR and `Δ`: bilinear vs. nearest-neighbour upsampling | Nearest-neighbour variant has a clearly lower peak |
| **H3c** | Without any spatial structure there is no prior | Peak PSNR of the MLP | Peak close to the noisy input PSNR (no real denoising) |
| **H4** | A good stopping point can be found without the clean image | PSNR lost compared with the oracle stop | Within ~1 dB of the oracle, and better than a fixed iteration budget |
| *H5 (optional)* | Averaging outputs over iterations (EMA, inspired by Cheng et al.) reduces sensitivity to the stopping point | PSNR of EMA output vs. plain output | Clearly smaller `Δ` |

**Which hypotheses are reproduced and which are new:**
- H1 and H2 reproduce findings of Ulyanov et al. (2018) and are presented as such.
- H3 and H4 are our own contribution.

**Prediction from the literature for H3:** H3b is confirmed, and H3a is weak or refuted. In other words, upsampling matters more than kernel size. Either outcome answers the research question: it tells us *which part* of the convolutional architecture is responsible.

**Scope of the claim:**
- "Responsible" in the research question is a causal claim.
- Our ablation provides evidence for which components are *necessary* in this setting: four images, Gaussian noise, one architecture family.
- It does not prove that they are sufficient, and it does not prove that the result generalises. We state this explicitly under Limitations.

---

## 4. Experimental setup

### Common settings
| Element | Choice |
|---|---|
| Images | 4 Kodak images (`kodim03`, `kodim05`, `kodim13`, `kodim15`), centre-cropped to 256×256 |
| Noise | Additive Gaussian, σ ∈ {25, 50} on the 0–255 scale, fixed seed, not clipped |
| Baseline model | Simplified U-Net from Ulyanov et al.: 4 levels, 64 channels, 4 skip channels, 3×3 convolutions, BatchNorm, LeakyReLU, bilinear upsampling, sigmoid output (≈ 454k parameters) |
| Input | Fixed random tensor `z` (32 channels), small perturbation (std 1/30) at each step |
| Optimisation | Adam, MSE against the noisy image, 3000 iterations |
| Logging | Every 10 iterations: training loss, held-out-pixel loss, PSNR and SSIM vs. clean, PSNR and SSIM of EMA output |
| Seeds | 1 seed per run; 3 seeds on one setting to show run-to-run variability |

### Experiments
| Experiment | Runs | Content |
|---|---|---|
| `lr_sweep` | 15 | 5 architectures × lr ∈ {0.001, 0.003, 0.01} on `kodim03`, σ = 25. Ensures a fair H3 comparison |
| `h1` | 10 | U-Net, 4 images × σ ∈ {25, 50}, plus 2 extra seeds |
| `h2` | 6 | U-Net fitted to clean image / shuffled pixels / uniform noise, 2 images |
| `h3` | 20 | 5 architectures (below) × 4 images, σ = 25 |
| `h4`, `h5` | 0 | Computed afterwards from the `h1` logs |

**Total:** 51 runs, about 2 minutes each on Apple MPS and faster on a Colab GPU.

### Architectures for H3
| Name | Kernels | Upsampling | Parameters | Tests |
|---|---|---|---|---|
| `unet` | 3×3 | bilinear | ≈ 454k | baseline |
| `unet_1x1` | 1×1 | bilinear | ≈ 69k | H3a |
| `unet_1x1_wide` | 1×1 (170 channels) | bilinear | ≈ 454k | H3a, controls for parameter count |
| `unet_nearest` | 3×3 | nearest-neighbour | ≈ 454k | H3b |
| `mlp` | none | none | ≈ 50.6M | H3c |

Parameter counts are reported with every result.

### Stopping rules (H4)
| Rule | Uses | Idea |
|---|---|---|
| Oracle | clean image | Iteration of maximum PSNR. Upper bound, not usable in practice |
| Fixed budget | — | Always stop at the same iteration (e.g. 1500) |
| Noise-level rule (discrepancy principle) | σ, true or estimated with `skimage.restoration.estimate_sigma` | Stop when the training MSE falls below σ². Any further decrease means the network is reproducing noise |
| Held-out pixels | — | 10 % of pixels are excluded from the loss. Stop at the minimum of the loss on those pixels |

### Evaluation metrics
| Metric | What it measures | Role |
|---|---|---|
| **PSNR** (dB) | Pixel-wise error: `10·log10(1/MSE)` | Primary metric, standard in the DIP literature, directly comparable with Ulyanov et al. |
| **SSIM** (0–1) | Similarity of local structure (luminance, contrast, correlation in small windows) | Secondary metric. Checks that conclusions don't depend on PSNR alone; closer to perceived quality, more sensitive to over-smoothing |
| **Visual comparison** | Zoomed crops of the same region | Shows what the numbers mean (blur vs. residual noise) |

**Checks with SSIM:**
- Does SSIM peak at the same iteration as PSNR (H1)?
- Do the H3 architectures rank the same under both metrics?
- In H4, report PSNR lost *and* SSIM lost against the oracle.

**Not used:** LPIPS. It needs a separate pretrained network, which would add an extra learned prior to an experiment about priors. Mention it under Limitations.

---

## 5. Repository

```
B4-Practical/
├── README.md
├── PLAN.md
├── pyproject.toml / poetry.lock     # Poetry environment (Python 3.11–3.12)
├── src/dip/
│   ├── utils.py                     # image IO, noise, PSNR, SSIM, seeding, device selection
│   ├── models.py                    # unet, unet_1x1, unet_1x1_wide, unet_nearest, mlp
│   ├── fit.py                       # DIP loop with logging, held-out pixels, EMA
│   └── stopping.py                  # TODO: stopping rules applied to logged runs
├── scripts/
│   ├── download_data.py             # Kodak images → data/processed/
│   └── run_all.py                   # runs experiments → results/<exp>/*.csv
├── notebooks/                       # one notebook per hypothesis; reads results/, writes figures/
├── results/  figures/
├── report/note.qmd + references.bib
└── papers/                          # literature PDFs
```

### Local setup (Poetry)
```bash
poetry env use python3.11
```
```bash
poetry install
```
```bash
poetry run python scripts/download_data.py
```
```bash
poetry run python scripts/run_all.py --exp h1
```
```bash
poetry run jupyter lab
```

`run_all.py` skips runs whose CSV already exists, so it can be interrupted and resumed. `--quick` runs a 100-iteration smoke test.

---

## 6. Running on Google Colab

Colab offers NVIDIA GPUs. The code picks CUDA automatically, so nothing has to change. Colab is most useful for the `h3` experiment, because the 50M-parameter MLP is slow on MPS, and for running experiments while working on the note locally.

**Before starting:** commit and push the repository to GitHub. If the repository is private, cloning requires a GitHub personal access token.

**1. Select a GPU runtime:** *Runtime → Change runtime type → GPU* (T4 on the free tier; L4/A100 with Colab Pro).

**2. Mount Google Drive.** The Colab disk is wiped when the session ends.
```python
from google.colab import drive
drive.mount('/content/drive')
```

**3. Get the code and install it.** Colab does not use Poetry. `pip` reads the same `pyproject.toml`.
```python
!git clone https://github.com/<account>/B4-Practical.git
%cd B4-Practical
!pip install -e .
!python scripts/download_data.py
```

**4. Keep results on Drive.** Results then survive disconnects, and resumed runs skip finished work.
```python
!mkdir -p /content/drive/MyDrive/B4_results
!rm -rf results && ln -s /content/drive/MyDrive/B4_results results
```

**5. Run the experiments:**
```python
!python scripts/run_all.py --exp lr_sweep
!python scripts/run_all.py --exp h1
!python scripts/run_all.py --exp h2
!python scripts/run_all.py --exp h3
```

**6. Bring the results back:** download `B4_results` from Google Drive and copy its contents into `results/` in the local repository. Then commit the CSV files so the figures can be regenerated by anyone.

**Notes:**
- If the session disconnects, rerun steps 2–5. Finished runs are skipped.
- Check the GPU with `!nvidia-smi`.
- If Colab's Python version moves outside the range set in `pyproject.toml` (3.11–3.12), `pip install -e .` will refuse to install. In that case, widen `requires-python`.

---

## 7. Figures

1. Clean / noisy / DIP at peak / DIP at final iteration, with a zoomed crop.
2. **H1:** PSNR and SSIM vs. iteration for σ = 25 and σ = 50.
3. **H2:** loss curves for natural image, shuffled pixels and noise (log scale).
4. **H3:** PSNR curves per architecture, plus a table of peak PSNR, peak SSIM, `Δ` and parameter count. Include zoomed crops of each architecture's best output.
5. **H4:** PSNR curve with each rule's stopping point marked, plus a table of PSNR and SSIM lost vs. the oracle.
6. *(optional)* **H5:** EMA vs. plain output.

---

## 8. Scientific note (Quarto, 3–4 pages)

| Section | Length |
|---|---|
| Problem and motivation | ½ page |
| Theoretical background | ½ page |
| Related literature | ¼ page |
| Experimental design | ½ page |
| Results | 1 page |
| Interpretation | ¼ page |
| Limitations | ¼ page |
| Conclusion | short |
| References | — |

Render with:
```bash
quarto render report/note.qmd --to pdf
```
This requires Quarto plus TinyTeX:
```bash
brew install --cask quarto
```
```bash
quarto install tinytex
```

---

## 9. Slides (~60 minutes)

| Minutes | Content |
|---|---|
| 0–8 | Motivation: image denoising; an untrained network as a prior |
| 8–20 | Theory: DIP formulation, why structure is fitted before noise, early stopping as regularisation |
| 20–27 | Literature: established results (Ulyanov, Cheng, Heckel & Hand, Wang et al.) |
| 27–32 | Research question, hypotheses, setup |
| 32–50 | Results H1–H4, each marked as reproduced or new |
| 50–57 | Interpretation, alternative explanations, limitations |
| 57–60 | Conclusion |

**Questions to prepare for:**
- Why is structure fitted before noise?
- What happens without early stopping?
- Is the effect caused by the convolution kernels or by the up-sampling?
- How do parameter counts affect the H3 comparison? (Answer: the `unet_1x1_wide` control.)
- Why use SSIM in addition to PSNR, and do they agree?
- Why does the noise-level rule need σ, and what happens when σ is misestimated?
- What does the experiment not show?

---

## 10. Literature

Every reference must be checked before citing. The PDFs are in `papers/`.

- Ulyanov, Vedaldi, Lempitsky (2018). *Deep Image Prior.* CVPR. arXiv:1711.10925. Seed paper.
- Cheng, Gadelha, Maji, Sheldon (2019). *A Bayesian Perspective on the Deep Image Prior.* CVPR. arXiv:1904.07457. Seed paper; motivates averaging (H5).
- Heckel & Hand (2019). *Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks.* ICLR. arXiv:1810.03982. Relevant to H3.
- Heckel & Soltanolkotabi (2020). *Denoising and Regularization via Exploiting the Structural Bias of Convolutional Generators.* ICLR. arXiv:1910.14634. Theory of why structure is fitted first; points to upsampling. Relevant to the theory section and H3b.
- Wang et al. (2023). *Early Stopping for Deep Image Prior.* TMLR. arXiv:2112.06074. Relevant to H4.
- Zhang et al. (2017). *Understanding Deep Learning Requires Rethinking Generalization.* ICLR. arXiv:1611.03530. Relevant to H2.
- Wang, Bovik, Sheikh, Simoncelli (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing. Defines SSIM. Not in `papers/`; cite only.

---

## 11. Status and next steps

**Done**
- [x] Repository structure and Poetry environment
- [x] Data download (4 Kodak images)
- [x] Models (`unet`, `unet_1x1`, `unet_1x1_wide`, `unet_nearest`, `mlp`), training loop with PSNR + SSIM logging, experiment runner
- [x] Baseline check: `kodim03`, σ = 25 gives noisy input 20.4 dB, peak 28.8 dB at iteration ~720, final 25.3 dB (EMA 27.7 dB)
- [x] Quarto note skeleton and bibliography

**Next**
1. Run `lr_sweep` and fix the learning rate per architecture in `run_all.py`.
2. Run `h1`, `h2`, `h3` (locally or on Colab).
3. Implement `src/dip/stopping.py` and evaluate H4 on the `h1` logs.
4. Build the notebooks and figures.
5. Write the Quarto note.
6. Prepare and rehearse the slides.

**Fallback if time runs short:** drop H5 first, then the estimated-σ variant in H4, then the 4th image.
