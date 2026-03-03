"""
Simple model copy for MiniCPM-SALA submission demo.

Usage:
    python preprocess_model.py --input /path/to/model --output /path/to/output
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in sorted(src.iterdir()):
        if f.name.startswith("."):
            continue
        target = dst / f.name
        if not target.exists():
            if f.is_dir():
                shutil.copytree(f, target)
            else:
                shutil.copy2(f, target)
            count += 1

    print(f"[preprocess] done — copied {count} files from {src} to {dst}")


if __name__ == "__main__":
    main()
