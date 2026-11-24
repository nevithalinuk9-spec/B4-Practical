# B4 · Architectural Priors in Image Restoration

**Research question.** Is the convolutional architecture itself responsible for Deep Image Prior (DIP) fitting image structure before noise, and can we choose when to stop without access to the clean image?

See [PLAN.md](PLAN.md) for the hypotheses and the experiment design, and [STUDY_GUIDE.md](STUDY_GUIDE.md) for a plain-language explanation of the method, the mathematics, and likely questions.

## Project structure

```
src/dip/          library code
  utils.py        image loading, noise, PSNR/SSIM, seeding, device selection
  models.py       UNet (3×3), 1×1 and nearest-upsampling variants, MLP
  fit.py          DIP optimisation loop with logging, held-out pixels and EMA
  stopping.py     stopping rules (oracle, fixed, noise level, held-out pixels) for H4
  analysis.py     loading logs, summary tables, figure style
scripts/
  download_data.py   downloads and crops the Kodak test images into data/
  run_all.py         runs the experiments, writes one CSV per run to results/
  make_figures.py    re-executes all notebooks → every figure and table
  build_notebooks.py regenerates the notebook files themselves (cells + written findings)
notebooks/        01_H1_peak, 02_H2_fitting_speed, 03_H3_architectures, 04_H4_H5_stopping
figures/          figures used in the note and slides
report/           Quarto scientific note (note.qmd → note.pdf) + references.bib
slides/           Quarto reveal.js slide deck (slides.qmd → slides.html)
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
poetry run python scripts/run_all.py --exp all
```
```bash
poetry run python scripts/make_figures.py
```

- `run_all.py --exp all` runs `h1`, `h2`, `lr_sweep` and `h3`: 56 runs, about 50 s each on an NVIDIA L4 and about 2.5 min on Apple MPS. `--exp h3_long` adds two 10,000-iteration runs of the 1×1 variants.
- `h3` automatically uses each model's best learning rate from `lr_sweep`.
- Runs whose CSV already exists are skipped, so the script can be interrupted and resumed. Use `--quick` for a short smoke test.
- The CSV logs from our runs are committed in `results/`, so `make_figures.py` alone reproduces every figure and table without re-running the experiments.
- To explore interactively, use `poetry run jupyter lab`.

### Note and slides

Quarto is installed as a Poetry dev dependency (`quarto-cli`). PDF output also needs TinyTeX, a one-time install:

```bash
poetry run quarto install tinytex
```
```bash
poetry run quarto render report/note.qmd --to pdf
```
```bash
poetry run quarto render slides/slides.qmd
```

- The note is `report/note.pdf` (3–4 pages plus references).
- The slides are a self-contained reveal.js page, `slides/slides.html`. Open it in a browser; press `S` for speaker notes, or append `?print-pdf` to the URL and print to get a PDF.

## Data

Four images from the Kodak Lossless True Color Image Suite (`kodim03`, `kodim05`, `kodim13`, `kodim15`), centre-cropped and resized to 256×256. Gaussian noise is added in code with a fixed seed, so the clean ground truth is always known.
