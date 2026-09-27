"""Pre-render real pipeline results for the web frontend's demo mode.

The Next.js site (web/) works without a backend by loading these files: every demo
image in demo/ is processed with all four combinations of the optional stages
(super-resolution, detection), exactly as the live API would return them.

Output (static files served by Next.js):
    web/public/demo/examples.json
    web/public/demo/report.json
    web/public/demo/<example>/<variant>/payload.json + images   (variant: sr0_det0, sr1_det0, ...)

Usage:
    python scripts/export_web_demo.py
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from irvision.api.payload import build_payload, file_writer, gray_u8, load_report, rgb_u8
from irvision.api.server import list_examples
from irvision.inference.io import read_image
from irvision.inference.pipeline import IRVisionPipeline
from irvision.utils.config import PROJECT_ROOT, load_config
from irvision.utils.log import get_logger

log = get_logger("export_web_demo")
OUT = PROJECT_ROOT / "web" / "public" / "demo"


def variant_id(sr: bool, det: bool) -> str:
    return f"sr{int(sr)}_det{int(det)}"


def main() -> None:
    cfg = load_config()
    pipeline = IRVisionPipeline(cfg)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    examples = []
    for ex in list_examples().values():
        image, nodata = read_image(ex["ir"])
        reference, _ = read_image(ex["reference"])
        write_thumb = file_writer(OUT / ex["id"], f"/demo/{ex['id']}")
        ref_f = np.moveaxis(reference.astype(np.float32) / 255.0, -1, 0)
        ir_f = image.astype(np.float32)
        lo, hi = np.percentile(ir_f[ir_f > 0], [2, 98])
        examples.append({
            "id": ex["id"], "title": ex["title"], "description": ex["description"],
            "thumbnail": write_thumb("thumb", rgb_u8(ref_f)[::2, ::2]),
            "ir_thumbnail": write_thumb("ir_thumb", gray_u8((ir_f - lo) / (hi - lo))[::2, ::2]),
        })
        for sr in (False, True):
            for det in (False, True):
                vid = variant_id(sr, det)
                result = pipeline.process_image(image, reference, nodata, super_resolution=sr, detection=det)
                ref_display = IRVisionPipeline.prepare_reference(reference, (3, *result.normalized.shape), [])
                payload = build_payload(result, file_writer(OUT / ex["id"] / vid, f"/demo/{ex['id']}/{vid}"),
                                        ref_display, sr, det, {"kind": "example", "id": ex["id"], "title": ex["title"]})
                (OUT / ex["id"] / vid / "payload.json").write_text(json.dumps(payload, indent=1))
                m = payload["metrics"]
                log.info("%-20s %s  PSNR %.2f  SSIM %.3f  agreement %s  %.0f ms", ex["id"], vid, m["psnr"], m["ssim"],
                         m.get("semantic", {}).get("agreement"), m["total_ms"])

    (OUT / "examples.json").write_text(json.dumps(examples, indent=1))
    (OUT / "report.json").write_text(json.dumps(load_report(cfg["paths"]["results_dir"]), indent=1))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    log.info("Wrote %s (%.1f MB)", OUT, size)


if __name__ == "__main__":
    main()
