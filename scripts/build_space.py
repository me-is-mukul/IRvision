"""Assemble a ready-to-push Hugging Face Space for the model server.

Two variants:
  --sdk gradio (default, free CPU Space): app.py starts the FastAPI server with uvicorn and
                a small status page; requirements.txt installs CPU PyTorch.
  --sdk docker: uses the repository's Dockerfile.

The trained models live in git-ignored outputs/, so the Space is built as a separate
folder containing exactly what the server needs:

    deploy/hf-space/
        README.md            Space settings
        .gitattributes       *.pt tracked with Git LFS
        app.py, requirements.txt, packages.txt        (gradio)
        Dockerfile, .dockerignore, requirements-api.txt (docker)
        irvision/  config/  demo/  outputs/...

Usage:
    python scripts/build_space.py                 # gradio variant
    python scripts/build_space.py --sdk docker
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from irvision.utils.config import PROJECT_ROOT

MODEL_FILES = [
    "outputs/models/unet_l1_ssim_9city/best.pt",
    "outputs/models/segmenter/best.pt",
    "outputs/models/baseline_lut.npy",
    "outputs/models/baseline_mean_color.npy",
    "outputs/pretrained/yolov8n-obb.pt",
]
DOCKER_FILES = ["Dockerfile", ".dockerignore", "requirements-api.txt", "pyproject.toml"]
DIRS = ["irvision", "config", "demo"]

README_HEADER = {
    "gradio": """---
title: IRVision API
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
pinned: false
---
""",
    "docker": """---
title: IRVision API
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
""",
}

README_BODY = """
# IRVision model server

FastAPI backend for the IRVision website: thermal-infrared satellite image enhancement,
colorization and semantic validation.

- `GET /api/health`: server status
- `GET /docs`: interactive API documentation
"""

APP_PY = '''"""Hugging Face Space entry point: the IRVision FastAPI server plus a small status page."""

import os

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr
import uvicorn

from irvision.api.server import app

with gr.Blocks(title="IRVision API") as status_page:
    gr.Markdown(
        "# IRVision model server\\n"
        "This Space serves the API used by the IRVision website.\\n\\n"
        "- Health check: [`/api/health`](/api/health)\\n"
        "- Interactive API docs: [`/docs`](/docs)"
    )

app = gr.mount_gradio_app(app, status_page, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
'''

GRADIO_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
torchvision==0.26.0+cpu
numpy>=1.26
opencv-python-headless>=4.8
rasterio>=1.3
scikit-image>=0.22
matplotlib>=3.8
PyYAML>=6.0
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
super-image>=0.2
ultralytics>=8.3
"""

PACKAGES_TXT = "libgl1\nlibglib2.0-0\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdk", choices=["gradio", "docker"], default="gradio")
    parser.add_argument("--gradio-version", default=None, help="default: the locally installed version")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "deploy" / "hf-space"))
    args = parser.parse_args()
    out = Path(args.out)

    needed = MODEL_FILES + (DOCKER_FILES if args.sdk == "docker" else [])
    missing = [f for f in needed if not (PROJECT_ROOT / f).exists()]
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
    for f in needed:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / f, out / f)
    (out / "outputs" / "results").mkdir(parents=True, exist_ok=True)
    for f in (PROJECT_ROOT / "outputs" / "results").glob("*_metrics.json"):
        shutil.copy2(f, out / "outputs" / "results" / f.name)

    if args.sdk == "gradio":
        version = args.gradio_version
        if version is None:
            try:
                import gradio

                version = gradio.__version__
            except ImportError:
                raise SystemExit("Pass --gradio-version (e.g. 5.x) or pip install gradio") from None
        header = README_HEADER["gradio"].format(gradio_version=version)
        (out / "app.py").write_text(APP_PY, encoding="utf-8")
        (out / "requirements.txt").write_text(GRADIO_REQUIREMENTS, encoding="utf-8")
        (out / "packages.txt").write_text(PACKAGES_TXT, encoding="utf-8")
    else:
        header = README_HEADER["docker"]

    (out / "README.md").write_text(header + README_BODY, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts) / 1e6
    print(f"{args.sdk} Space folder ready: {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
