"""Phase 5: train the U-Net colorizer.

Uses the `model` and `training` sections of config/config.yaml. Command-line
flags override single values so experiments don't require editing the config.

Usage:
    python scripts/train.py --run-name unet_l1
    python scripts/train.py --run-name unet_l1_ssim --ssim 1.0
    python scripts/train.py --run-name smoke --epochs 1        # quick check that everything runs

Outputs: outputs/models/<run-name>/{best.pt,last.pt,history.csv,curves.png,summary.json}
To use a run in the app/pipeline, set `inference.checkpoint` in the config.
"""

from __future__ import annotations

import argparse

from irvision.training.trainer import train
from irvision.utils.config import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", required=True, help="folder name under outputs/models/")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--lr", type=float, help="override training.learning_rate")
    parser.add_argument("--l1", type=float, help="override training.loss_weights.l1")
    parser.add_argument("--ssim", type=float, help="override training.loss_weights.ssim")
    parser.add_argument("--perceptual", type=float, help="override training.loss_weights.perceptual (VGG19)")
    parser.add_argument("--inputs", nargs="+", help="override model.inputs, e.g. ir_clahe ir_abs")
    parser.add_argument("--arch", choices=["unet", "resunet34"], help="override model.name")
    parser.add_argument("--aux", type=float, help="weight of the land-cover head loss (resunet34); 0 = off")
    parser.add_argument("--batch-size", type=int, help="override training.batch_size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg["training"]
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.lr is not None:
        t["learning_rate"] = args.lr
    if args.l1 is not None:
        t["loss_weights"]["l1"] = args.l1
    if args.ssim is not None:
        t["loss_weights"]["ssim"] = args.ssim
    if args.perceptual is not None:
        t["loss_weights"]["perceptual"] = args.perceptual
    if args.inputs:
        cfg["model"]["inputs"] = args.inputs
    if args.arch:
        cfg["model"]["name"] = args.arch
    if args.aux is not None:
        t["loss_weights"]["aux"] = args.aux
    if args.batch_size:
        t["batch_size"] = args.batch_size

    train(cfg, args.run_name, get_device(cfg))


if __name__ == "__main__":
    main()
