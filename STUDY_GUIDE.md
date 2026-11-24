# Study Guide — Deep Image Prior, Explained Simply

This guide explains everything behind our project, from "what is an image" up to the stopping rules. It ends with likely examiner questions and short answers. Read it top to bottom once, then use section 12 to test yourselves.

> **One-sentence summary:** We let an untrained neural network copy one noisy photo. It copies the real picture before the noise, so if we stop at the right time we get a cleaned photo. We test *which part of the network* causes this, and *how to know when to stop* without seeing the clean photo.

---

## 1. Images are just numbers

- A colour image of 256 × 256 pixels has 3 numbers per pixel: red, green, blue.
- We scale every number to the range **0 to 1** (0 = black, 1 = full colour).
- So one image is a big table of **256 × 256 × 3 = 196,608 numbers**.
- In the maths we treat this table as one long vector `x`.

---

## 2. Noise

We add noise ourselves:

$$y = x + n$$

| Symbol | Meaning |
|---|---|
| `x` | clean image (we know it, because we made the experiment) |
| `n` | noise: one random number per pixel and channel |
| `y` | noisy image (the only thing the network sees) |

**Gaussian noise** `n ~ N(0, σ²)`:
- Each noise value is drawn independently from a bell curve centred at 0.
- `σ` (sigma) is the typical size of the noise.
- We use σ = 25 and σ = 50 on the 0–255 scale. On our 0–1 scale that is 25/255 ≈ **0.098** and 50/255 ≈ **0.196**.
- **Variance** σ² is the average of n². For σ = 25: σ² ≈ **0.0096**. This number comes back in the stopping rule.

**Independent** means the noise at one pixel tells you *nothing* about the noise at any other pixel. This is the key property the whole project relies on.

---

## 3. Measuring quality: MSE, PSNR, SSIM

### MSE (mean squared error)
$$\text{MSE}(a, b) = \frac{1}{N}\sum_{i=1}^{N} (a_i - b_i)^2$$
Take the difference per number, square it (so negatives don't cancel), and average. 0 means identical.

### PSNR (peak signal-to-noise ratio)
$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{1}{\text{MSE}}\right) \quad \text{[dB]}$$
- The "1" is the maximum pixel value.
- Higher is better.
- **+3 dB ≈ half the error. +10 dB = ten times less error.**
- Our numbers: noisy input ≈ 20 dB, good DIP result ≈ 29 dB.

### SSIM (structural similarity)
- A score from 0 to 1 (1 = identical).
- It compares **small windows** of the two images on three things: average brightness, contrast, and whether the patterns go up and down together (correlation).
- Why we use it: PSNR treats every pixel error the same. SSIM cares about **structure**, so it is closer to what a human sees. It reacts more strongly to blur and to lost texture.
- We use it as a **second opinion**: if PSNR and SSIM agree, our conclusions are more trustworthy.

---

## 4. Neural network building blocks (only what we use)

### Convolution (the core idea)
A 3×3 convolution looks at a **3×3 window** of pixels and computes one output number:

$$\text{out}(i,j) = \sum_{a=-1}^{1}\sum_{b=-1}^{1} w_{a,b}\cdot \text{in}(i+a,\, j+b) + \text{bias}$$

- The same 9 weights `w` are used **at every position** of the image. This is called **weight sharing**.
- Consequences:
  - The network treats every location the same way. This is useful because a pattern (an edge, a texture) looks the same wherever it appears.
  - Every output pixel depends on its **neighbours**. That makes it easy to produce smooth, structured content.
- With many channels, each output channel is a weighted sum over all input channels' windows.
- **Parameters of one conv layer:** `k · k · c_in · c_out + c_out`. For example, 3×3, 64 → 64 channels gives 36,928 weights.

### 1×1 convolution
- The window is only the pixel itself: it mixes **channels**, but never **neighbours**.
- This is our `unet_1x1` variant. It removes the spatial part of convolution.

### Strided convolution (downsampling)
- A convolution with **stride 2** moves its window two pixels at a time, so the output is half the size (256 → 128).
- This lets deeper layers "see" larger areas of the image.

### Upsampling
Makes a small image bigger (16 → 32 → … → 256).
- **Bilinear:** a new pixel is a weighted **average** of its nearest neighbours. This produces smooth results. It is our baseline.
- **Nearest-neighbour:** a new pixel is a **copy** of the closest old pixel. This produces blocky results and adds no smoothing. It is our `unet_nearest` variant.

### BatchNorm
- For each channel, subtract the mean and divide by the standard deviation, then apply a learned scale and shift.
- It keeps the numbers in a healthy range so training is stable.

### LeakyReLU (the nonlinearity)
$$\text{LeakyReLU}(v) = \begin{cases} v & v > 0 \\ 0.2\,v & v \le 0 \end{cases}$$
- Without a nonlinearity, any stack of layers would collapse into one linear function and couldn't draw complex images.

### Sigmoid (output)
$$\text{sigmoid}(v) = \frac{1}{1+e^{-v}}$$
- It squeezes any number into (0, 1), so the output is always a valid colour.

### Fully connected layer (used in the MLP)
- Every output is connected to every input with its own weight.
- There is **no notion of neighbours** at all.

---

## 5. Our U-Net (the baseline network)

```
z (32 × 256 × 256)
 │──skip──────────────────────────────────────────────┐
 ▼ down: stride-2 conv + conv   → 64 × 128 × 128      │
 │──skip─────────────────────────────────────┐        │
 ▼ down                          → 64 × 64 × 64       │        │
 │──skip────────────────────────┐            │        │
 ▼ down                          → 64 × 32 × 32       │        │
 │──skip──────────┐             │            │        │
 ▼ down                          → 64 × 16 × 16       │        │
 ▼ up (bilinear) → 32, concat skip, conv 3×3, conv 1×1 │        │
 ▼ up → 64 ──────────────────────┘            │        │
 ▼ up → 128 ──────────────────────────────────┘        │
 ▼ up → 256 ───────────────────────────────────────────┘
 ▼ conv 1×1 → 3 channels → sigmoid → image (3 × 256 × 256)
```

- **Encoder (down path):** shrinks the image 4 times. At the bottom (16 × 16) each value describes a large part of the image, which is where large-scale shapes are formed.
- **Decoder (up path):** grows the image back to full size.
- **Skip connections:** at each level, 4 channels are passed directly from the down path to the up path, so fine details are not lost.
- **Input `z`:** 32 channels of random numbers (uniform, × 0.1). It is created once and **never changes**. It's just a fixed "canvas".
- **Input perturbation:** at each step we add tiny noise (std 1/30) to `z`. This acts as extra regularisation and comes from the original paper.
- **Size:** ≈ 454,000 parameters.

---

## 6. How the network is "trained"

There's no dataset. We fit the network to **one** noisy image.

### The loss
$$L(\theta) = \frac{1}{N}\sum_{i} \big(f_\theta(z)_i - y_i\big)^2$$
- `θ` (theta) = all the weights of the network.
- `f_θ(z)` = the image the network draws.
- `L` measures how different the drawing is from the **noisy** photo.

### Gradient descent
The loss is like the height of a hill. We want to walk downhill:
$$\theta \leftarrow \theta - \text{lr}\cdot\nabla_\theta L$$
- `∇L` (the gradient) points uphill, so we step the other way.
- `lr` (learning rate, here 0.01) is the step size.
- PyTorch computes `∇L` automatically (backpropagation).

### Adam
- An improved gradient descent:
  - It keeps a running average of past gradients (**momentum**, smoother direction).
  - It scales each weight's step by how large its gradients usually are (**adaptive step size**).
- It's the standard choice and the one used in the DIP paper.

### What happens over time
- If we keep going long enough, the network can reproduce `y` **exactly, noise included**. It has 454k parameters to fit 196k numbers.
- But **on the way**, it passes through a point where it has drawn the image and not yet the noise. **That point is our result.**

Our baseline run (kodim03, σ = 25):

| Iteration | PSNR vs. clean |
|---|---|
| 0 (noisy input) | 20.4 dB |
| ~720 (peak) | **28.8 dB** |
| 3000 (end) | 25.3 dB |

---

## 7. Why the image comes before the noise (the core theory)

### Frequencies, intuitively
- Any image can be written as a sum of waves (this is the Fourier transform).
- **Low frequency** = slow, wide waves: smooth areas, overall shapes, colour gradients.
- **High frequency** = fast, tiny waves: fine detail, sharp edges and **noise**.
- **Natural images** have most of their energy in **low** frequencies: large areas look similar.
- **White noise** has its energy **spread equally over all frequencies**. Since there are many more high-frequency waves than low ones, most of the noise lives in high frequencies.

### Spectral bias
- Neural networks trained with gradient descent learn **low frequencies first** and high frequencies much later. This is called **spectral bias**.
- For convolutional generators it is especially strong:
  - **Weight sharing + small kernels:** the easiest things to produce are patterns that repeat and change smoothly.
  - **Bilinear upsampling:** it literally averages neighbours, which is a low-pass (smoothing) filter.
- Heckel & Soltanolkotabi (2020) prove, for a simplified convolutional generator, that it fits the low-frequency part of the signal faster, and that the **upsampling** is a key reason.

### Putting it together
- The image is mostly low frequency, so it is fitted **fast**.
- The noise is mostly high frequency, so it is fitted **slowly**.
- That gap in speed is the whole trick. Early stopping exploits it.

### Experiment H2 checks this directly
- **Natural image** as the target: fitted quickly.
- **Shuffled pixels:** the same colours, but the structure is destroyed. If this is fitted slowly, the speed comes from *structure*, not from the colours.
- **Pure noise:** fitted slowest.

---

## 8. What we change in H3 (the architecture ablation)

| Model | What's different | What we learn |
|---|---|---|
| `unet` | baseline | — |
| `unet_1x1` | all kernels 1×1, so no neighbour mixing in convolutions | Do we need spatial kernels? (H3a) |
| `unet_1x1_wide` | same, but 170 channels, so parameters match the baseline (≈ 454k) | Is `unet_1x1` worse only because it is smaller? |
| `unet_nearest` | nearest-neighbour instead of bilinear upsampling | Do we need smooth upsampling? (H3b) |
| `mlp` | fully connected, no spatial structure at all | Is there any prior without structure? (H3c) |

**What the literature predicts:** upsampling matters more than kernel size (Heckel & Hand's Deep Decoder works with only 1×1 convolutions plus upsampling).

**Fair-comparison measures:**
- **Learning-rate sweep:** each architecture gets its best learning rate, so none loses just because of a bad step size.
- **Parameter-matched variant:** `unet_1x1_wide` has the same number of parameters as the baseline.
- **Same everything else:** same images, noise, iterations and seeds.

**Note on the MLP:** it has 50.6M parameters, almost all in the last layer (256 hidden units × 196,608 outputs). Each output pixel has its **own personal weights**, so nothing links a pixel to its neighbours.

---

## 9. Stopping without the clean image (H4)

In real life we don't have `x`, so we can't watch PSNR. We need a signal computed only from `y`.

### Rule A — Oracle (not usable, just the upper bound)
Stop at the iteration with the highest PSNR. We report everything else as "**dB lost compared with the oracle**".

### Rule B — Fixed budget
Always stop at the same iteration, e.g. 1500. It's simple, but the best iteration differs per image and noise level.

### Rule C — Noise-level rule (discrepancy principle)
**Idea:** if the network had drawn the *perfect* clean image `x`, its training loss would be:
$$\frac{1}{N}\lVert x - y\rVert^2 = \frac{1}{N}\lVert n \rVert^2 \approx \sigma^2$$
- So a loss **below σ²** means the network is now matching some of the noise. **Stop when the loss first falls below σ².**
- It needs σ. We test two versions: the true σ, and σ **estimated from the noisy image** with `skimage.restoration.estimate_sigma`.
- **Weak point:** if σ is estimated too high, we stop too early (blurry result). If it's estimated too low, we stop too late (noisy result).

### Rule D — Held-out pixels (self-validation)
**Idea:** hide 10 % of the pixels from the loss (like test questions that were never shown during studying). Watch the error on those hidden pixels and stop at its minimum.

**Why it works (the one derivation worth knowing):**
On the hidden pixels `M`, write the error against the noisy image as:
$$f - y = (f - x) - n$$
Square it and take the average over the random noise:
$$\mathbb{E}\lVert f - y\rVert_M^2 = \lVert f - x\rVert_M^2 \;-\; 2\,\mathbb{E}\langle f - x,\, n\rangle_M \;+\; \mathbb{E}\lVert n\rVert_M^2$$
- The network **never saw** `n` on `M`. It was trained only on other pixels, whose noise is independent. So `f` doesn't depend on `n_M`, and the middle term is **0**.
- The last term is just `|M| · σ²`, a **constant**.

Therefore:
$$\underbrace{\mathbb{E}\lVert f - y\rVert_M^2}_{\text{what we can measure}} = \underbrace{\lVert f - x\rVert_M^2}_{\text{true error (unknown)}} + \underbrace{|M|\sigma^2}_{\text{constant}}$$

The measurable held-out loss is the true error **on the held-out pixels** plus a constant, so they have their minimum at the same iteration. We find that minimum **without ever seeing `x`**, and without knowing σ.

- **Cost:** the network sees 10 % fewer pixels. It can still fill them in from their neighbours, which is exactly the prior we are studying. Measured cost: about 0.2 dB of peak PSNR.
- **Practical issue:** the held-out loss is noisy from step to step, so smooth it (moving average) before taking the minimum.
- **What we found:** this rule stops **too late** (about 1 dB lost on average). The catch is in the bold words above: the error on held-out pixels is not the error on the whole image. Held-out pixels are never fitted directly; they are only filled in from their neighbours. So the noise reaches them later, and their error keeps falling after the full-image error has started to rise.

### H5 (optional) — EMA averaging
$$\text{avg}_t = 0.99\cdot\text{avg}_{t-1} + 0.01\cdot f_t$$
- This is the average over roughly the last 100 outputs (1 / (1 − 0.99) = 100).
- **The idea:** random step-to-step fluctuations cancel out, so the result might be less sensitive to *when* we stop.
- **What we found:** the decline after the peak is *not* smaller (3.3 vs. 3.1 dB), so H5 as stated is refuted. But the EMA output is about **2 dB better at every point** (mean peak 27.2 vs. 25.1 dB). A likely reason: each plain output comes from a randomly perturbed input `z`, so it is jittery. Averaging ~100 outputs works like an ensemble.
- It's inspired by Cheng et al., who average posterior samples using SGLD, a more principled Bayesian method. **Ours is a simplification, not their method.**

---

## 10. The literature in one line each

| Paper | Main point | Used for |
|---|---|---|
| Ulyanov et al. 2018, *Deep Image Prior* | An untrained CNN is a strong image prior; early stopping is essential | Foundation; H1, H2 |
| Cheng et al. 2019, *Bayesian Perspective on DIP* | DIP ≈ Gaussian process at infinite width; posterior averaging (SGLD) avoids early stopping | Theory; H5 |
| Heckel & Hand 2019, *Deep Decoder* | 1×1 convolutions + upsampling, few parameters: still a strong prior, and doesn't overfit | H3 prediction |
| Heckel & Soltanolkotabi 2020, *Structural Bias of Convolutional Generators* | Proof that convolutional generators fit low frequencies first; upsampling is key | Theory; H3b |
| Wang et al. 2023, *Early Stopping for DIP* | Practical stopping rule based on the variance of recent outputs | H4 context |
| Zhang et al. 2017, *Rethinking Generalization* | Networks can memorise even random labels; they just take longer | H2 context |
| Wang et al. 2004, *SSIM* | Defines SSIM | Metric |

---

## 10b. Our results on one page

Details, tables and figures are in `notebooks/` and `figures/`. The noisy input at σ = 25 scores about 20.2–20.4 dB.

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| **H1** PSNR peaks, then falls | ✅ confirmed | Every run peaks. Mean drop 1.6 dB (σ = 25) vs. 4.5 dB (σ = 50). Seed spread ≈ 0.1 dB |
| **H2** structure fitted before noise | ✅ confirmed (reproduces Ulyanov) | Natural image below MSE 0.01 after 20–130 iterations; shuffled pixels after 2700 or never; noise never |
| **H3a** 3×3 kernels needed | ✅ within a normal budget; mainly a *speed* effect | At 3000 iterations: mean peak 26.4 dB → 20.1 dB (1×1) → 22.1 dB (1×1, same #params). With 10,000 iterations the matched 1×1 halves its gap (27.5 vs. 29.1 dB on kodim03) but fits >15× more slowly |
| **H3b** bilinear upsampling needed | ❌ for the peak, but it matters for the decline | Same peak (26.6 dB), but reached ~3× earlier and dropping 5.2 vs. 1.4 dB afterwards |
| **H3c** no prior without spatial structure | ✅ confirmed | MLP ≈ noisy input (+0.3 dB); memorises the noise within 500 iterations |
| **H4** stopping without the clean image | ✅ confirmed | Noise-level rule with *estimated* σ: 0.24 dB lost on average. Held-out pixels: 1.0 dB. Fixed 1500: 1.6 dB |
| **H5** EMA makes stopping less critical | ❌ refuted as stated | Drop unchanged (3.3 vs. 3.1 dB), but the whole curve is ~2 dB higher |

**Answer to the research question, in two sentences:**
- The convolutional architecture is responsible, but different parts do different jobs. The **3×3 spatial kernels** make structure fit fast, more than 15× faster than without them. That's why there is little or no denoising without them within a normal budget, even with equal parameters, and none at all without any spatial structure. **Bilinear upsampling** slows down the later fitting of noise.
- And **yes, we can stop without the clean image**: the discrepancy principle with σ estimated from the noisy image lands within about 0.25 dB of the best possible stopping point.

**Important caveat:** the 10,000-iteration runs show that the 1×1 networks are mostly *slower*, not hopeless. The matched one halves its gap and is still improving. Whether it would eventually match the U-Net is open, so say "kernels are necessary *within a normal iteration budget*".

---

## 11. What our experiment can and cannot show

**Can show:**
- Whether PSNR peaks and falls (H1).
- Whether structure is fitted faster than noise (H2).
- Which architectural parts are *necessary* for the effect in our setting (H3).
- How close no-reference stopping rules get to the oracle (H4).

**Cannot show / limitations:**
- Only 4 images, 256 × 256, synthetic Gaussian noise. Real camera noise is different.
- Mostly 1 seed per run, so small differences (< ~0.5 dB) may be random.
- "Necessary" is not "sufficient": removing a part and losing the effect shows that the part matters, not that it alone explains everything.
- One architecture family (U-Net style). Other designs could behave differently.
- PSNR and SSIM are imperfect proxies for human perception. LPIPS was not used, because it relies on a pretrained network.
- The learning-rate sweep is done on one image only.

---

## 12. Likely questions from the professor, with short answers

### About the method
**Q1. What is the Deep Image Prior in one sentence?**
An untrained, randomly initialised CNN is fitted to a single corrupted image. Because it reproduces natural image structure faster than noise, stopping early gives a restored image.

**Q2. Where is the training data?**
There is none. The only data is the one noisy image. The "prior" comes from the architecture, not from learned examples.

**Q3. What is a "prior"?**
An assumption about what solutions are likely before seeing the data. Here: "images look like what this network produces easily", i.e. smooth, structured content.

**Q4. Why does the network eventually fit the noise?**
It is over-parameterised (≈ 454k parameters for ≈ 196k pixel values), so with enough steps it can reproduce any target, including noise.

**Q5. Why do we need early stopping?**
Without it, the output converges to the noisy image. Stopping is the regularisation: it decides how much of the high-frequency content (mostly noise) we accept.

**Q6. What does the fixed input `z` do? Could we use a real image instead?**
It's just a fixed random code that the network transforms into the output. Its content doesn't matter. All the structure comes from the network and the optimisation.

### About the theory
**Q7. Why is image structure fitted before noise?**
Spectral bias. Convolutional generators fit low frequencies first. Natural images are mostly low frequency, white noise is spread across all frequencies, so the image comes first.

**Q8. What is spectral bias?**
The tendency of networks trained by gradient descent to learn smooth, low-frequency functions before high-frequency ones.

**Q9. What does the Bayesian paper add?**
It shows that in the infinite-width limit DIP behaves like a Gaussian process prior over images. Averaging posterior samples instead of picking one stopping point makes results less sensitive to when you stop.

**Q10. Why might the upsampling matter more than the kernels?**
Bilinear upsampling averages neighbouring pixels. That is a smoothing (low-pass) operation built into the network. The Deep Decoder works with only 1×1 convolutions plus upsampling, which suggests the upsampling carries much of the prior.

### About the design
**Q11. Why add noise synthetically instead of using real noisy photos?**
So that we know the clean image and can measure PSNR and SSIM exactly, including the oracle stopping point.

**Q12. Why shuffle pixels in H2?**
Shuffling keeps the colour statistics but destroys the spatial structure. If the shuffled image is fitted as slowly as noise, then speed depends on structure, not on colours.

**Q13. How do you make sure the H3 comparison is fair?**
Three measures:
- each architecture gets its own best learning rate (sweep)
- a parameter-matched 1×1 variant
- identical images, noise, iterations and seeds

**Q14. Why is the parameter-matched variant important?**
Otherwise a worse 1×1 network could simply be explained by "it has 7× fewer parameters", not by "it lacks spatial kernels".

**Q15. Why not 3 seeds everywhere?**
Compute and time. We run 3 seeds on one setting to estimate run-to-run variation and treat smaller differences with caution.

**Q16. Why not clip the noisy image to [0, 1]?**
Clipping would change the noise distribution (it would no longer be exactly Gaussian with variance σ²). The noise-level stopping rule relies on that variance.

### About stopping
**Q17. Why does the held-out-pixel rule work?**
The network never sees the noise on held-out pixels, so the expected held-out loss equals the true error plus a constant (σ² per entry). Both have their minimum at the same iteration.

**Q18. What is the discrepancy principle?**
Stop when the fit to the data is as good as the noise level allows, i.e. training MSE ≈ σ². Fitting better than that means fitting noise.

**Q19. What if σ is unknown or estimated wrongly?**
Overestimating σ stops too early (blurry output). Underestimating it stops too late (noisy output). The held-out rule doesn't need σ, which is its advantage.

**Q20. Isn't hiding 10 % of the pixels a disadvantage?**
Slightly, because less data is used. But the network fills those pixels in from their neighbours, which is exactly the prior we study. The final image still has values at every pixel.

### About results and interpretation
**Q21. What did you reproduce, and what is new?**
- **Reproduced:** the PSNR peak (H1) and fast fitting of natural images (H2) from Ulyanov et al.
- **New:** the architecture ablation separating kernels from upsampling (H3), and the comparison of stopping rules against the oracle (H4).

**Q22. Why use SSIM as well as PSNR?**
PSNR measures average pixel error. SSIM measures local structure and is closer to perceived quality. If both agree, conclusions don't depend on one metric.

**Q23. What would refute your main claim?**
- If the 1×1 and nearest-upsampling variants denoise just as well as the baseline, the specific convolutional components aren't responsible.
- If the MLP also showed a clear peak, the effect wouldn't need spatial structure at all.

**Q24. What does your experiment *not* show?**
It doesn't show:
- that the results generalise to real noise, other image sizes or other architectures
- that the components are sufficient

It shows only which components are necessary in this setting.

**Q25. How would you extend this with more time?**
- More images and seeds.
- Real camera noise.
- Inpainting.
- A frequency-band analysis showing directly when each frequency is fitted.
- SGLD as in Cheng et al.

### About reproducibility
**Q26. How can someone reproduce your results?**
1. `poetry install`
2. `scripts/download_data.py`
3. `scripts/run_all.py --exp <name>`
4. Run the notebooks to regenerate the figures from the CSV logs.

Seeds are fixed for initialisation, noise and the held-out mask.

**Q27. Why does a rerun give slightly different numbers?**
GPU and MPS operations are not perfectly deterministic. Differences are small. That's one reason we report a 3-seed spread.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Adam | Gradient-descent variant with momentum and adaptive step sizes |
| Backpropagation | Algorithm that computes the gradient of the loss for every weight |
| BatchNorm | Normalises each channel to mean 0 / std 1, then learns a scale and shift |
| Discrepancy principle | Stop when the data fit reaches the noise level σ² |
| EMA | Exponential moving average of outputs over iterations |
| Early stopping | Ending the optimisation before convergence, as a form of regularisation |
| Inductive bias / prior | What a model "prefers" before seeing data |
| Learning rate | Step size of gradient descent |
| Over-parameterised | More parameters than needed to fit the data exactly |
| PSNR | Pixel error on a log scale, in dB; higher is better |
| Regularisation | Anything that prevents fitting noise |
| Skip connection | Shortcut passing features from the encoder to the decoder |
| Spectral bias | Networks learn low frequencies before high frequencies |
| SSIM | Structural similarity, 0–1; compares local brightness, contrast and patterns |
| Stride | How far a convolution window moves per step; stride 2 halves the size |
| Upsampling | Enlarging a feature map (bilinear = smooth, nearest = blocky) |
| Weight sharing | The same convolution weights are used at every position |
