"""Download the Kodak test images and save 256×256 centre crops to data/processed/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests
from skimage import io

from dip.utils import DATA_DIR, center_crop_resize

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/{name}.png"
IMAGES = ["kodim03", "kodim05", "kodim13", "kodim15"]
SIZE = 256


def main() -> None:
    raw_dir = DATA_DIR.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in IMAGES:
        raw_path = raw_dir / f"{name}.png"
        if not raw_path.exists():
            print(f"downloading {name} ...")
            r = requests.get(KODAK_URL.format(name=name), timeout=60)
            r.raise_for_status()
            raw_path.write_bytes(r.content)

        img = center_crop_resize(io.imread(raw_path)[..., :3], SIZE)
        out_path = DATA_DIR / f"{name}.png"
        io.imsave(out_path, (img * 255).round().astype(np.uint8), check_contrast=False)
        print(f"saved {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}")


if __name__ == "__main__":
    main()
