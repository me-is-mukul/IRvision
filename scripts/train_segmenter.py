"""Phase 7: fine-tune the land-cover segmenter used for semantic validation.

Needs patches with `landcover` labels (scripts/download_landcover.py, then
scripts/prepare_dataset.py). Settings: `semantic` section of config.yaml.

Usage:
    python scripts/train_segmenter.py                 # -> outputs/models/segmenter/
    python scripts/train_segmenter.py --epochs 1      # quick check
"""

from __future__ import annotations

import argparse

from irvision.semantic.train import train_segmenter
from irvision.utils.config import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="segmenter")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["semantic"]["epochs"] = args.epochs
    summary = train_segmenter(cfg, args.run_name, get_device(cfg))
    print(summary)


if __name__ == "__main__":
    main()
