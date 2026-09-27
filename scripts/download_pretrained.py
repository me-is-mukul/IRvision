"""Download the pre-trained models used by the advanced features into outputs/pretrained/.

    EDSR-base x2     super-resolution (super-image, DIV2K photos; Hugging Face cache) ~5 MB
    yolov8n-obb.pt   YOLOv8n oriented-box detector trained on DOTA aerial imagery   ~6 MB
    VGG19            ImageNet weights for the perceptual loss (torchvision cache)    ~550 MB

None of these are trained by this project (PLAN.md §8). Safe to re-run: existing files are skipped.

Usage:
    python scripts/download_pretrained.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from irvision.utils.config import load_config, resolve_path
from irvision.utils.log import get_logger

log = get_logger("download_pretrained")


def main() -> None:
    cfg = load_config()
    opt = cfg["optional"]
    from irvision.models.super_resolution import SuperResolver

    SuperResolver.load(opt["super_resolution"]["model"], opt["super_resolution"]["scale"])
    log.info("EDSR x%d weights ready", opt["super_resolution"]["scale"])

    yolo_path = resolve_path(opt["detection"]["model_path"])
    if not yolo_path.exists():
        from ultralytics import YOLO

        yolo_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(yolo_path.name)                      # ultralytics downloads it into the CWD
        src = Path(model.ckpt_path)
        shutil.move(str(src), yolo_path)
        log.info("downloaded %s", yolo_path.name)
    else:
        log.info("%s exists, skipping", yolo_path.name)

    from torchvision.models import VGG19_Weights, vgg19

    vgg19(weights=VGG19_Weights.IMAGENET1K_V1)             # cached in ~/.cache/torch
    log.info("VGG19 weights ready")


if __name__ == "__main__":
    main()
