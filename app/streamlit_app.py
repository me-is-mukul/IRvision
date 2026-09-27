"""IRVision dashboard.

Run from the repository root:
    streamlit run app/streamlit_app.py

This file only does UI. All processing goes through
``irvision.inference.pipeline.IRVisionPipeline.process_image`` so the app and
the scripts can never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from irvision.inference.io import read_image, to_png_bytes
from irvision.inference.pipeline import IRVisionPipeline, load_example_scene
from irvision.models.baseline import ColormapColorizer, LUTColorizer
from irvision.semantic.landcover import CLASSES, PALETTE, colorize_labels
from irvision.utils.config import load_config

st.set_page_config(page_title="IRVision", page_icon="🛰️", layout="wide")

CFG = load_config()
PATHS = CFG["paths"]
COLORIZERS = {
    "U-Net (trained model)": "unet",
    "LUT baseline (no learning)": "lut",
    "Pseudo-colour (inferno)": "colormap",
}


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading model ...")
def get_pipeline(kind: str) -> IRVisionPipeline:
    """One pipeline per colorizer, loaded once per server process."""
    if kind == "unet":
        return IRVisionPipeline(CFG)          # U-Net, or its documented fallback chain
    if kind == "lut":
        lut_path = PATHS["models_dir"] / "baseline_lut.npy"
        if lut_path.exists():
            return IRVisionPipeline(CFG, colorizer=LUTColorizer.load(lut_path))
    return IRVisionPipeline(CFG, colorizer=ColormapColorizer(CFG["baseline"]["colormap"]))


@st.cache_data(show_spinner="Loading scene ...")
def get_example(scene_dir: str):
    return load_example_scene(scene_dir, CFG)


def example_scenes() -> dict[str, Path]:
    """Scenes available as demo inputs, held-out (never trained on) first."""
    scenes = {}
    raw = PATHS["raw_dir"]
    if not raw.exists():
        return scenes
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        meta_path = d / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        # older downloads have no "aoi" field: match their stored point against the config AOIs
        aoi = meta.get("aoi") or next((a["name"] for a in CFG["dataset"]["download"]["aois"]
                                       if a["point"] == meta.get("aoi_point_lonlat")), "scene")
        label = f"{aoi} - {d.name[10:25]}" + ("  (held-out, never seen in training)" if meta.get("holdout") else "  (training scene)")
        scenes[label] = d
    return dict(sorted(scenes.items(), key=lambda kv: "held-out" not in kv[0]))


def draw_detections(img_hwc: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw oriented boxes and labels (yellow) on an (H, W, 3) float image."""
    canvas = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8).copy()
    thickness = max(1, canvas.shape[0] // 400)
    for d in detections:
        pts = np.array(d["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (255, 220, 0), thickness)
        x, y = int(d["box"][0]), max(10, int(d["box"][1]) - 3)
        cv2.putText(canvas, f"{d['class_name']} {d['confidence']:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35 * thickness, (255, 220, 0), thickness)
    return canvas.astype(np.float32) / 255.0


def gamma(img_chw: np.ndarray, g: float) -> np.ndarray:
    """(3, H, W) -> (H, W, 3) with display gamma."""
    return np.clip(np.moveaxis(img_chw, 0, -1), 0, 1) ** (1.0 / g)


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🛰️ IRVision")
source = st.sidebar.radio("Input", ["Example Landsat scene", "Upload an IR image"])

image, reference, nodata, input_label = None, None, None, ""
if source == "Example Landsat scene":
    scenes = example_scenes()
    if not scenes:
        st.sidebar.warning("No scenes in data/raw. Run scripts/download_landsat.py first.")
    else:
        label = st.sidebar.selectbox("Scene", list(scenes))
        crop = st.sidebar.select_slider("Crop size (px, 30 m each)", [256, 512, 1024, 2048], value=512)
        ir_full, rgb_full, _ = get_example(str(scenes[label]))
        h, w = ir_full.shape
        crop = min(crop, h, w)
        cx = st.sidebar.slider("Crop position x", 0, w - crop, (w - crop) // 2, step=32)
        cy = st.sidebar.slider("Crop position y", 0, h - crop, (h - crop) // 2, step=32)
        image = ir_full[cy : cy + crop, cx : cx + crop]
        reference = rgb_full[:, cy : cy + crop, cx : cx + crop]
        input_label = f"{label.split('  ')[0]}, crop {crop}px at ({cx}, {cy})"
        if "training scene" in label:
            st.sidebar.caption("⚠️ Training scene: only its bottom 15% (test stripe) is unseen. "
                               "Use the held-out scene for honest metrics.")
else:
    up = st.sidebar.file_uploader("IR image (single band)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    ref_up = st.sidebar.file_uploader("Optional: true-colour reference (for PSNR/SSIM)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    st.sidebar.caption("GeoTIFF (e.g. Landsat *_ST_B10.TIF), 8/16-bit PNG or JPEG. Values may be in any unit.")
    if up is not None:
        try:
            image, nodata = read_image(up.getvalue(), up.name)
            input_label = up.name
            if ref_up is not None:
                reference, _ = read_image(ref_up.getvalue(), ref_up.name)
                if reference.dtype == np.uint16:
                    reference = reference.astype(np.float32) / 65535.0
        except Exception as exc:  # show the error in the UI instead of crashing the app
            st.sidebar.error(f"Could not read the file: {exc}")
            image = None

model_label = st.sidebar.selectbox("Colorizer", list(COLORIZERS))
display_gamma = st.sidebar.slider("Display brightness (gamma)", 1.0, 2.5, 1.8, 0.1,
                                  help="Only changes how RGB images are shown, not the metrics.")
st.sidebar.markdown("**Advanced stages** (off by default, see *Model report*)")
use_sr = st.sidebar.toggle(
    "Super-resolution ×2 (EDSR)", value=CFG["optional"]["super_resolution"]["enabled"],
    help="Pre-trained EDSR upsamples the IR before colorization. Measured on the test set it LOWERS "
         "PSNR, SSIM and semantic agreement: the U-Net was trained on 30 m pixels.")
use_det = st.sidebar.toggle(
    "Object detection (YOLOv8-OBB, aerial)", value=CFG["optional"]["detection"]["enabled"],
    help="Pre-trained on DOTA aerial imagery (0.1–1 m/px). At Landsat's 30 m most objects are sub-pixel. "
         "On colorized images it produced only false 'plane' detections in testing.")
run = st.sidebar.button("▶ Process", type="primary", disabled=image is None, width="stretch")

# --------------------------------------------------------------------------- header
st.title("IRVision: infrared → RGB for satellite images")
st.caption("Landsat 8/9 thermal Band 10 → CLAHE enhancement → U-Net colorization → metrics. "
           "The colour image is a **learned, plausible reconstruction**, not the true-colour photo.")

tab_result, tab_report, tab_about = st.tabs(["Result", "Model report", "How it works"])

if run and image is not None:
    pipeline = get_pipeline(COLORIZERS[model_label])
    try:
        with st.spinner("Processing ..."):
            st.session_state["result"] = pipeline.process_image(image, reference, nodata,
                                                                super_resolution=use_sr, detection=use_det)
            st.session_state["meta"] = {"input": input_label, "model": model_label, "info": pipeline.colorizer_info,
                                        "has_ref": reference is not None, "reference": reference}
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(f"Input rejected: {exc}")

# --------------------------------------------------------------------------- result tab
with tab_result:
    res, meta = st.session_state.get("result"), st.session_state.get("meta")
    if res is None:
        st.info("Choose an input in the sidebar and press **Process**. "
                "Tip: the held-out Hyderabad scene was never used in training and has ground truth.")
    else:
        st.markdown(f"**Input:** {meta['input']}  ·  **Colorizer:** {meta['model']}")
        for w in res.warnings:
            st.warning(w)

        t = res.metrics["time_ms"]
        m = st.columns(4)
        m[0].metric("PSNR (dB)", f"{res.metrics['psnr']:.2f}" if "psnr" in res.metrics else "n/a",
                    help="vs. true-colour reference, valid pixels only. Needs a reference image.")
        m[1].metric("SSIM", f"{res.metrics['ssim']:.3f}" if "ssim" in res.metrics else "n/a")
        m[2].metric("Inference time", f"{t['total']:.0f} ms", help="Whole pipeline, this image")
        m[3].metric("Colorization time", f"{t['colorization']:.0f} ms")

        cols = st.columns(4 if meta["has_ref"] else 3)
        cols[0].image(res.normalized, caption="Original IR (contrast-stretched)", clamp=True, width="stretch")
        cols[1].image(res.enhanced, caption="Enhanced IR (CLAHE): model input", clamp=True, width="stretch")
        colorized_view = gamma(res.colorized, display_gamma)
        if res.detections:
            colorized_view = draw_detections(colorized_view, res.detections)
        sr_note = f", super-resolved ×{res.scale}" if res.scale > 1 else ""
        cols[2].image(colorized_view, caption=f"Colorized RGB ({res.colorizer}{sr_note})", width="stretch")
        if meta["has_ref"]:
            ref = np.nan_to_num(np.asarray(meta["reference"], dtype=np.float32))
            if ref.ndim == 3 and ref.shape[0] == 3:
                cols[3].image(gamma(ref, display_gamma), caption="Ground truth RGB", width="stretch")

        st.subheader("Semantic map (land cover)")
        if res.semantic_map is None:
            st.info("No semantic map for this run. See the warning above for the reason.")
        else:
            sem = res.metrics.get("semantic")
            if sem:
                s1, s2, s3 = st.columns(3)
                s1.metric("Semantic agreement", f"{sem['pixel_agreement']:.1%}",
                          help="Share of pixels where the land-cover class of the colorized image "
                               "equals that of the true image.")
                s2.metric("mIoU (colorized vs true)", f"{sem['miou']:.3f}")
                s3.metric("Mean Dice", f"{sem['mean_dice']:.3f}")
            m1, m2, m3 = st.columns([2, 2, 1])
            m1.image(colorize_labels(res.semantic_map), caption="Land cover seen in the colorized RGB", width="stretch")
            if res.reference_semantic_map is not None:
                m2.image(colorize_labels(res.reference_semantic_map), caption="Land cover seen in the true RGB",
                         width="stretch")
            m3.markdown("**Legend**")
            for name, colour in zip(CLASSES, PALETTE):
                m3.markdown(f'<span style="display:inline-block;width:12px;height:12px;background:rgb{tuple(int(c) for c in colour)};'
                            f'margin-right:6px;border:1px solid #888"></span>{name}', unsafe_allow_html=True)
            m3.caption("Segmenter: DeepLabV3 fine-tuned on real Landsat RGB → ESA WorldCover classes.")

        if res.detections is not None:
            st.subheader("Object detection")
            det_m = res.metrics.get("detection")
            d1, d2, d3 = st.columns(3)
            d1.metric("Objects in colorized image", len(res.detections))
            if res.reference_detections is not None:
                d2.metric("Objects in true image", len(res.reference_detections))
            if det_m:
                d3.metric("Matched (same class, IoU ≥ 0.5)", det_m["tp"])
            if res.detections:
                st.dataframe(pd.DataFrame([{"class": d["class_name"], "confidence": d["confidence"],
                                            "box (x1, y1, x2, y2)": d["box"]} for d in res.detections]),
                             hide_index=True)
            st.caption("YOLOv8n-OBB pre-trained on DOTA aerial imagery. At 30 m/pixel, treat detections as "
                       "unverified; on colorized images they were all false in testing (Model report).")

        st.download_button("⬇ Download colorized PNG", to_png_bytes(res.colorized ** (1 / display_gamma)),
                           file_name="irvision_colorized.png", mime="image/png")
        with st.expander("Details: stage timings, model, normalization"):
            st.dataframe(pd.DataFrame({"stage": list(t), "ms": [round(v, 1) for v in t.values()]}),
                         hide_index=True)
            st.json({"model": meta["info"], "normalization": res.metrics["normalization"],
                     "input_shape": res.metrics["input_shape"], "valid_fraction": round(res.metrics["valid_fraction"], 4)})

# --------------------------------------------------------------------------- report tab
with tab_report:
    st.markdown("Numbers below are read from files written by the evaluation scripts. Nothing here is typed in by hand.")
    results_dir = PATHS["results_dir"]
    model_json = results_dir / "model_metrics.json"
    base_json = results_dir / "baseline_metrics.json"
    if model_json.exists():
        rep = json.loads(model_json.read_text())
        st.subheader(f"Test results ({rep['split']} split, {rep['num_patches']} patches)")
        st.dataframe(pd.DataFrame(rep["table"]), hide_index=True)
        if rep.get("holdout_table"):
            st.subheader("Held-out scene only (never seen in training)")
            st.dataframe(pd.DataFrame(rep["holdout_table"]), hide_index=True)
        st.caption(rep.get("metric_notes", ""))
        fig = results_dir / "model_comparison.png"
        if fig.exists():
            st.image(str(fig), caption="IR input | LUT baseline | U-Net | ground truth")
    elif base_json.exists():
        st.info("No U-Net evaluation yet. Showing baselines only (run scripts/evaluate_model.py).")
        st.json(json.loads(base_json.read_text())["methods"])
    else:
        st.info("No evaluation results yet. Run scripts/evaluate_baseline.py and scripts/evaluate_model.py.")

    sem_json = results_dir / "semantic_metrics.json"
    if sem_json.exists():
        sem = json.loads(sem_json.read_text())
        st.subheader("Semantic validation: does the colorized image mean the same thing?")
        rows = []
        for subset in ("all", "holdout"):
            for key, r in sem["results"][subset].items():
                rows.append({"test patches": subset, "comparison": key.replace("_", " "),
                             "mIoU": round(r["miou"], 3), "mean Dice": round(r["mean_dice"], 3),
                             "pixel agreement": round(r["pixel_agreement"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("'true vs worldcover' is the ceiling: how well the segmenter itself works on real RGB. "
                   "'X vs true' is the semantic consistency of method X. " + sem.get("notes", ""))
        fig = results_dir / "semantic_comparison.png"
        if fig.exists():
            st.image(str(fig), caption="True RGB | its land cover | U-Net RGB | its land cover | WorldCover labels")

    adv_json = results_dir / "advanced_metrics.json"
    if adv_json.exists():
        adv = json.loads(adv_json.read_text())
        st.subheader("Advanced stages: do they help?")
        sr_rows = [{"test patches": s, "method": m.replace("unet_sr", "U-Net + EDSR x2").replace("unet", "U-Net"),
                    "PSNR": r["psnr"], "SSIM": r["ssim"], "semantic mIoU": r["semantic_miou"],
                    "semantic agreement": r["semantic_agreement"]}
                   for s, res_s in adv["super_resolution"]["results"].items() for m, r in res_s.items()]
        st.markdown("**Super-resolution** (scored at the original 30 m resolution)")
        st.dataframe(pd.DataFrame(sr_rows), hide_index=True)
        det_rows = [{"test patches": s, "colorizer": m, "objects found": r["n_pred"], "objects in true RGB": r["n_ref"],
                     "matched": r["tp"], "precision": r["precision"], "recall": r["recall"]}
                    for s, res_s in adv["detection"]["results"].items() for m, r in res_s.items()]
        st.markdown("**Object detection** (reference = detections on the true RGB; no object ground truth exists)")
        st.dataframe(pd.DataFrame(det_rows), hide_index=True)
        st.caption(f"Classes found on true RGB: {adv['detection']['classes_on_true_rgb']} · "
                   f"on U-Net RGB: {adv['detection']['classes_on_unet_rgb']}")

    curves = PATHS["models_dir"] / Path(CFG["inference"]["checkpoint"]).parent.name / "curves.png"
    if curves.exists():
        st.image(str(curves), caption="Training curves of the deployed model")

# --------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
**Pipeline** (`irvision.inference.pipeline.process_image`)
1. **Validation**: single band, size limits, no-data pixels masked.
2. **Normalization**: 2–98 percentile stretch to [0, 1] (works for Kelvin, DN, 8/16-bit).
3. **CLAHE**: local contrast enhancement of the IR.
4. **Colorization**: U-Net (1 → 3 channels) trained on Landsat 8/9 Band 10 → Bands 4/3/2.
5. **Semantic map**: a land-cover segmenter (DeepLabV3, fine-tuned on real Landsat RGB with
   ESA WorldCover labels) classifies the colorized image. With a reference, it also classifies the
   true image, and the two maps are compared (agreement, mIoU, Dice).
6. **Metrics**: PSNR / SSIM against a reference if given, and per-stage timing.

**Advanced, optional** (sidebar toggles; off by default because measurements showed no benefit):
- **Super-resolution ×2**: pre-trained EDSR before colorization.
- **Object detection**: pre-trained YOLOv8-OBB (DOTA aerial imagery) on the colorized image.
- The colorizer itself can be trained with a **VGG19 perceptual loss** (`scripts/train.py --perceptual`).
See README → Evaluation & results and Design decisions (D-019, D-020, D-022).

**Data**: Landsat 9 Collection-2 Level-2 scenes (Bengaluru, Delhi, Nile Delta for training;
Hyderabad held out). **Limits**: the thermal band is natively 100 m, so fine detail is inferred,
not observed. The output shows plausible colours, not measured ones.
""")
