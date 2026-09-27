"""Assemble a ready-to-push Hugging Face Space (Docker) for the model server.

The trained models live in git-ignored outputs/, so the Space is built as a separate
folder containing exactly what the Dockerfile needs:

    deploy/hf-space/
        README.md            Space settings (Docker SDK, port 7860)
        .gitattributes       *.pt tracked with Git LFS
        Dockerfile, .dockerignore, requirements-api.txt, pyproject.toml
        irvision/  config/  demo/
        outputs/models/...   outputs/results/*.json   outputs/pretrained/yolov8n-obb.pt

Usage:
    python scripts/build_space.py
    python scripts/build_space.py --out some/other/folder
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from irvision.utils.config import PROJECT_ROOT

FILES = [
    "Dockerfile", ".dockerignore", "requirements-api.txt", "pyproject.toml",
    "outputs/models/unet_l1_ssim_9city/best.pt",
    "outputs/models/segmenter/best.pt",
    "outputs/models/baseline_lut.npy",
    "outputs/models/baseline_mean_color.npy",
    "outputs/pretrained/yolov8n-obb.pt",
]
DIRS = ["irvision", "config", "demo"]

SPACE_README = """---
title: IRVision API
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# IRVision model server

FastAPI backend for the IRVision website: thermal-infrared satellite image enhancement,
colorization and semantic validation.

- `GET /api/health`: server status
- `GET /docs`: interactive API documentation
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "deploy" / "hf-space"))
    args = parser.parse_args()
    out = Path(args.out)

    missing = [f for f in FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Missing files (train / download them first): {missing}")

    # keep an existing .git folder so the Space can be updated in place
    if out.exists():
        for item in out.iterdir():
            if item.name != ".git":
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    out.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in DIRS:
        shutil.copytree(PROJECT_ROOT / d, out / d, ignore=ignore)
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / f, out / f)
    (out / "outputs" / "results").mkdir(parents=True, exist_ok=True)
    for f in (PROJECT_ROOT / "outputs" / "results").glob("*_metrics.json"):
        shutil.copy2(f, out / "outputs" / "results" / f.name)

    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts) / 1e6
    print(f"Space folder ready: {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
