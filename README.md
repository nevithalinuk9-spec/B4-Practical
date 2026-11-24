# B4 · Architectural Priors in Image Restoration

**Research question.** Is the convolutional architecture itself responsible for Deep Image Prior (DIP) fitting image structure before noise, and can we choose when to stop without access to the clean image?

See [PLAN.md](PLAN.md) for the hypotheses and the experiment design, and [STUDY_GUIDE.md](STUDY_GUIDE.md) for a plain-language explanation of the method, the mathematics, and likely questions.

## Project structure

```
src/dip/          library code
  utils.py        image loading, noise, PSNR/SSIM, seeding, device selection
  models.py       UNet (3×3), 1×1 and nearest-upsampling variants, MLP
  fit.py          DIP optimisation loop with logging, held-out pixels and EMA
scripts/
  download_data.py   downloads and crops the Kodak test images into data/
  run_all.py         runs the experiments, writes one CSV per run to results/
notebooks/        one notebook per hypothesis; loads results/ and makes figures
figures/          figures used in the note and slides
report/           Quarto scientific note (note.qmd) + references.bib
papers/           seed papers and further literature
```

## Setup

Requires Python 3.11 or 3.12 and [Poetry](https://python-poetry.org/) 2.x.

```bash
poetry env use python3.11
```
```bash
poetry install
```

On Google Colab (no Poetry), clone the repo and run `pip install -e .` instead.

## Reproduce

```bash
poetry run python scripts/download_data.py
```
```bash
poetry run python scripts/run_all.py --exp h1
```
```bash
poetry run jupyter lab
```

`run_all.py` skips runs whose CSV already exists, so it can be interrupted and resumed. Use `--quick` for a short smoke test (few iterations, one image).

Render the note (needs [Quarto](https://quarto.org) and `quarto install tinytex`):

```bash
quarto render report/note.qmd --to pdf
```

## Data

Four images from the Kodak Lossless True Color Image Suite (`kodim03`, `kodim05`, `kodim13`, `kodim15`), centre-cropped and resized to 256×256. Gaussian noise is added in code with a fixed seed, so the clean ground truth is always known.
