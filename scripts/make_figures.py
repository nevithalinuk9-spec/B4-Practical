"""Re-execute all analysis notebooks, which regenerates every figure in figures/ and every table.

    poetry run python scripts/make_figures.py

Needs the CSV logs in results/ (produced by scripts/run_all.py).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def main() -> None:
    for path in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")):
        print(f"executing {path.name} ...")
        nb = nbformat.read(path, as_version=4)
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NOTEBOOKS)}})
        nbformat.write(nb, path)
    print("done: figures are in figures/")


if __name__ == "__main__":
    main()
