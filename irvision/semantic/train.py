"""Fine-tune the land-cover segmenter on *true* Landsat RGB -> WorldCover labels.

Trained on the train split only, selected by val mIoU. It is a measuring
instrument for semantic validation, so it never sees colorized images.

Outputs in outputs/models/<run_name>/: best.pt, history.csv, summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from irvision.semantic.landcover import CLASSES, IGNORE
from irvision.semantic.metrics import confusion_matrix, summarize
from irvision.semantic.segmenter import build_segmenter, normalize_input
from irvision.training.augment import RandomFlipRotate
from irvision.training.trainer import set_seed
from irvision.utils.log import get_logger

log = get_logger(__name__)


class LandCoverPatches(Dataset):
    """(rgb (3,S,S) float32, labels (S,S) int64 with IGNORE on invalid pixels), cached in RAM."""

    def __init__(self, split_dir: str | Path, augment: RandomFlipRotate | None = None):
        files = sorted(Path(split_dir).glob("*.npz"))
        self.items = []
        for f in files:
            with np.load(f) as d:
                if "landcover" not in d:
                    raise KeyError(f"{f.name} has no landcover; run scripts/download_landcover.py + prepare_dataset.py")
                # extra channels are stored (1, S, S); labels are (S, S)
                labels = np.where(d["valid"], d["landcover"][0], IGNORE).astype(np.uint8)
                self.items.append((d["rgb"], labels))
        if not self.items:
            raise FileNotFoundError(f"No patches in {split_dir}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        rgb, labels = self.items[i]
        rgb, labels = rgb.astype(np.float32), labels[None].astype(np.float32)
        if self.augment:
            rgb, labels, _ = self.augment(rgb, labels, labels)
        return torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(np.ascontiguousarray(labels[0])).long()


def class_weights(ds: LandCoverPatches) -> torch.Tensor:
    """Median-frequency balancing so rare classes (water, bare) are not ignored."""
    counts = np.zeros(len(CLASSES))
    for _, labels in ds.items:
        counts += np.bincount(labels[labels != IGNORE], minlength=len(CLASSES))
    freq = counts / counts.sum()
    w = np.median(freq[freq > 0]) / np.maximum(freq, 1e-6)
    return torch.tensor(np.clip(w, 0.1, 10.0), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for rgb, labels in loader:
        pred = model(normalize_input(rgb.to(device)))["out"].argmax(1).cpu().numpy().astype(np.uint8)
        lab = labels.numpy()
        lab = np.where(lab == IGNORE, IGNORE, lab).astype(np.uint8)
        cm += confusion_matrix(pred, lab)
    return summarize(cm)


def train_segmenter(cfg: dict, run_name: str, device: str) -> dict:
    scfg, paths = cfg["semantic"], cfg["paths"]
    set_seed(cfg["project"]["seed"])
    run_dir = paths["models_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LandCoverPatches(paths["train_dir"], RandomFlipRotate(cfg["project"]["seed"]))
    val_ds = LandCoverPatches(paths["val_dir"])
    train_dl = DataLoader(train_ds, batch_size=scfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=scfg["batch_size"])
    weights = class_weights(train_ds).to(device)
    log.info("Segmenter %s: %d train / %d val patches, class weights %s", run_name, len(train_ds), len(val_ds),
             dict(zip(CLASSES, weights.cpu().numpy().round(2).tolist())))

    model = build_segmenter(len(CLASSES), pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["learning_rate"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=scfg["epochs"])
    best, history = -1.0, []
    for epoch in range(1, scfg["epochs"] + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for rgb, labels in train_dl:
            rgb, labels = rgb.to(device), labels.to(device)
            logits = model(normalize_input(rgb))["out"]
            loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_miou": val["miou"],
               "val_pixel_agreement": val["pixel_agreement"], "seconds": time.perf_counter() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        improved = val["miou"] > best
        if improved:
            best = val["miou"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "epoch": epoch, "val": val}, run_dir / "best.pt")
        log.info("epoch %2d | loss %.4f | val mIoU %.4f acc %.4f | %.1fs%s", epoch, row["train_loss"], val["miou"],
                 val["pixel_agreement"], row["seconds"], "  *best*" if improved else "")

    best_ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"run_name": run_name, "best_epoch": best_ckpt["epoch"], "best_val": best_ckpt["val"],
               "epochs_run": len(history), "train_minutes": round(sum(h["seconds"] for h in history) / 60, 2)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
