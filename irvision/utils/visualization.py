"""Figures for checking data by eye. All functions save a PNG and return its path."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: works in scripts, tests and servers
import matplotlib.pyplot as plt
import numpy as np


def to_display_rgb(rgb_chw: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) for imshow. Gamma is for display only."""
    img = np.clip(np.nan_to_num(np.moveaxis(rgb_chw, 0, -1)), 0, 1)
    return img ** (1.0 / gamma) if gamma != 1.0 else img


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


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 128) -> np.ndarray:
    """Interleave two (H, W, 3) images in a checkerboard — misalignment shows as broken edges."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // tile + xx // tile) % 2 == 0)[..., None]
    return np.where(mask, a, b)


def edge_overlay(ir: np.ndarray, rgb_hwc: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """RGB image with IR edges drawn in magenta."""
    ir8 = (np.clip(np.nan_to_num(ir), 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(ir8, (0, 0), 2), 30, 80) > 0
    edges &= valid
    out = rgb_hwc.copy()
    out[edges] = (1.0, 0.0, 1.0)
    return out


def save_scene_overview(
    path: str | Path,
    scene_id: str,
    ir: np.ndarray,
    ir_clahe: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    alignment_text: str = "",
    crop: int | None = 512,
    gamma: float = 1.8,
) -> Path:
    """Six-panel figure: IR, CLAHE IR, RGB, validity mask, checkerboard and edge overlay.

    The bottom row uses a centre crop of size ``crop`` so misalignment is visible.
    """
    rgb_disp = to_display_rgb(rgb, gamma)
    ir_rgb = np.repeat(np.clip(ir_clahe, 0, 1)[..., None], 3, axis=-1)

    h, w = valid.shape
    c = crop or min(h, w)
    r0, c0 = max(0, (h - c) // 2), max(0, (w - c) // 2)
    win = np.s_[r0 : r0 + c, c0 : c0 + c]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (ir, "IR Band 10 (normalized)", "inferno"),
        (ir_clahe, "IR + CLAHE", "gray"),
        (rgb_disp, "RGB B4/B3/B2 (display gamma)", None),
        (valid, f"Valid pixels ({100 * valid.mean():.1f}%)", "gray"),
        (checkerboard(ir_rgb[win], rgb_disp[win], tile=max(16, c // 8)), "Checkerboard IR/RGB (centre crop)", None),
        (edge_overlay(ir_clahe[win], rgb_disp[win], valid[win]), "IR edges on RGB (centre crop)", None),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{scene_id}\n{alignment_text}", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def save_patch_grid(
    path: str | Path,
    ir_patches: Sequence[np.ndarray],
    rgb_patches: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    gamma: float = 1.8,
) -> Path:
    """Two rows per column: IR patch (1,S,S) above its RGB target (3,S,S)."""
    n = len(ir_patches)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(ir_patches[i][0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(to_display_rgb(rgb_patches[i], gamma))
        if titles:
            axes[0, i].set_title(titles[i], fontsize=7)
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].set_ylabel("IR")
    fig.tight_layout()
    return _save(fig, path)


def save_comparison_grid(
    path: str | Path,
    rows: Sequence[Sequence[np.ndarray]],
    col_titles: Sequence[str],
    row_titles: Sequence[str] | None = None,
    gamma: float = 1.8,
) -> Path:
    """Grid of images: one row per sample, one column per method.

    Each image is (H, W) grayscale or (3, H, W) RGB in [0, 1]. The same display
    gamma is applied to every RGB panel so methods are compared fairly.
    """
    n_rows, n_cols = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows + 0.4), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(to_display_rgb(img, gamma))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0 and row_titles:
                ax.set_ylabel(row_titles[r], fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
