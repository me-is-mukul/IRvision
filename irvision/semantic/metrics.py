"""Agreement between two label maps: IoU per class, mIoU, Dice, pixel agreement.

Accumulate a confusion matrix over many images with ``confusion_matrix`` +
addition, then summarize once with ``summarize`` (the standard way to compute
dataset-level mIoU).
"""

from __future__ import annotations

import numpy as np

from irvision.semantic.landcover import CLASSES, IGNORE


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = len(CLASSES),
                     valid: np.ndarray | None = None) -> np.ndarray:
    """(num_classes x num_classes) counts, rows = target, cols = pred. IGNORE pixels skipped."""
    keep = (pred != IGNORE) & (target != IGNORE)
    if valid is not None:
        keep &= valid
    idx = target[keep].astype(np.int64) * num_classes + pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def summarize(cm: np.ndarray, class_names: list[str] = CLASSES) -> dict:
    """IoU and Dice per class, their means over classes that occur, and pixel agreement.

    A class counts toward the mean only if it appears in the target or the
    prediction; classes absent from both are reported as ``None``.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    present = (tp + fp + fn) > 0
    iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
    dice = np.where(present, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    total = cm.sum()
    return {
        "miou": float(np.nanmean(iou)) if present.any() else float("nan"),
        "mean_dice": float(np.nanmean(dice)) if present.any() else float("nan"),
        "pixel_agreement": float(tp.sum() / total) if total else float("nan"),
        "iou": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(class_names, iou)},
        "dice": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(class_names, dice)},
        "pixels": int(total),
    }


def compare_maps(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """``summarize(confusion_matrix(...))`` for a single pair of maps."""
    return summarize(confusion_matrix(pred, target, valid=valid))
