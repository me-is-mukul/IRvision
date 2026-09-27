"""Consistency between two sets of detections (colorized image vs true image).

The detections on the true RGB act as reference: a detection on the colorized
image is a true positive if it has the same class and box IoU >= ``iou_threshold``
with a not-yet-matched reference detection (greedy, highest confidence first).
"""

from __future__ import annotations


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two axis-aligned boxes (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_counts(pred: list[dict], ref: list[dict], iou_threshold: float = 0.5) -> dict:
    """True positives, false positives, false negatives of ``pred`` against ``ref``."""
    used = [False] * len(ref)
    tp = 0
    for p in sorted(pred, key=lambda d: -d["confidence"]):
        best, best_iou = -1, iou_threshold
        for j, r in enumerate(ref):
            if not used[j] and r["class_name"] == p["class_name"]:
                iou = box_iou(p["box"], r["box"])
                if iou >= best_iou:
                    best, best_iou = j, iou
        if best >= 0:
            used[best] = True
            tp += 1
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(ref) - tp, "n_pred": len(pred), "n_ref": len(ref)}


def summarize_counts(counts: dict) -> dict:
    """Precision / recall / F1 from accumulated counts. ``None`` when undefined (nothing to divide)."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if tp + fp + fn else None)
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def compare_detections(pred: list[dict], ref: list[dict], iou_threshold: float = 0.5) -> dict:
    return summarize_counts(match_counts(pred, ref, iou_threshold))
