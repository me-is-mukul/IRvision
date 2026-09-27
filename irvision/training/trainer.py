"""Training loop for the U-Net colorizer.

Everything is driven by the ``model`` and ``training`` config sections. The
checkpoint with the best validation score is kept; the test split is never
touched here.

Outputs in ``outputs/models/<run_name>/``:
    best.pt          weights + config + validation metrics (load with models.unet.load_checkpoint)
    last.pt          weights after the final epoch
    history.csv      one row per epoch (rewritten every epoch: watch it for live progress)
    curves.png       loss / PSNR / SSIM curves
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from irvision.models.unet import build_unet
from irvision.training.augment import RandomFlipRotate
from irvision.training.dataset import PatchDataset
from irvision.training.losses import ColorizationLoss, masked_ssim
from irvision.utils.log import get_logger

log = get_logger(__name__)


@dataclass
class TrainResult:
    run_dir: Path
    best_epoch: int
    best_metrics: dict
    history: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked PSNR per image (B,), same definition as evaluation.metrics.psnr."""
    sq = ((pred.float() - target.float()) ** 2 * valid).sum(dim=(1, 2, 3))
    mse = sq / (valid.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp_min(1.0)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def validate(model, loader, loss_fn, device, autocast_dtype) -> dict[str, float]:
    model.eval()
    losses, psnrs, ssims = [], [], []
    for batch in loader:
        x, y, v = (batch[k].to(device, non_blocking=True) for k in ("input", "target", "valid"))
        with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype, enabled=autocast_dtype is not None):
            pred = model(x)
        pred = pred.float()
        losses.append(loss_fn(pred, y, v)[0].item())
        psnrs.extend(batch_psnr(pred, y, v).tolist())
        ssims.extend(masked_ssim(pred[i : i + 1], y[i : i + 1], v[i : i + 1]).item() for i in range(len(pred)))
    return {"val_loss": float(np.mean(losses)), "val_psnr": float(np.mean(psnrs)), "val_ssim": float(np.mean(ssims))}


def train(cfg: dict, run_name: str, device: str) -> TrainResult:
    tcfg, mcfg, paths = cfg["training"], cfg["model"], cfg["paths"]
    set_seed(cfg["project"]["seed"])
    inputs = mcfg["inputs"]
    run_dir = paths["models_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PatchDataset(paths["train_dir"], inputs, transform=RandomFlipRotate(cfg["project"]["seed"]), cache=True)
    val_ds = PatchDataset(paths["val_dir"], inputs, cache=True)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=tcfg["num_workers"], pin_memory=device.startswith("cuda"))
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"])
    log.info("Run %s: %d train / %d val patches, inputs=%s, device=%s", run_name, len(train_ds), len(val_ds), inputs, device)

    model = build_unet(mcfg, in_channels=len(inputs)).to(device)
    log.info("U-Net parameters: %.2f M", sum(p.numel() for p in model.parameters()) / 1e6)
    loss_fn = ColorizationLoss(**tcfg["loss_weights"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    # bfloat16 autocast on GPUs that support it: ~2x faster, no loss scaling needed
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else None

    select = tcfg["select_metric"]                  # val_psnr or val_ssim (higher is better)
    best_score, best_epoch, best_metrics, stale = -np.inf, -1, {}, 0
    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, train_losses = time.perf_counter(), []
        for batch in train_dl:
            x, y, v = (batch[k].to(device, non_blocking=True) for k in ("input", "target", "valid"))
            with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred = model(x)
            loss, _ = loss_fn(pred.float(), y, v)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        metrics = validate(model, val_dl, loss_fn, device, autocast_dtype)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)   # live progress, survives crashes

        improved = metrics[select] > best_score
        if improved:
            best_score, best_epoch, best_metrics, stale = metrics[select], epoch, metrics, 0
            _save(run_dir / "best.pt", model, cfg, epoch, metrics)
        else:
            stale += 1
        log.info("epoch %3d | train %.4f | val loss %.4f psnr %.2f ssim %.4f | %.1fs%s", epoch, row["train_loss"],
                 metrics["val_loss"], metrics["val_psnr"], metrics["val_ssim"], row["seconds"], "  *best*" if improved else "")
        if stale >= tcfg["early_stopping_patience"]:
            log.info("Early stopping: no %s improvement for %d epochs", select, stale)
            break

    _save(run_dir / "last.pt", model, cfg, epoch, metrics)
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    _plot_curves(hist, run_dir / "curves.png")
    (run_dir / "summary.json").write_text(json.dumps(
        {"run_name": run_name, "best_epoch": best_epoch, "select_metric": select, "best_val": best_metrics,
         "epochs_run": len(hist), "inputs": inputs, "loss_weights": tcfg["loss_weights"],
         "train_minutes": round(hist["seconds"].sum() / 60, 2)}, indent=2))
    log.info("Best epoch %d: %s", best_epoch, {k: round(v, 4) for k, v in best_metrics.items()})
    return TrainResult(run_dir, best_epoch, best_metrics, hist)


def _save(path: Path, model, cfg: dict, epoch: int, metrics: dict) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "model_cfg": cfg["model"],
        "inputs": cfg["model"]["inputs"],
        "preprocessing": cfg["preprocessing"],
        "loss_weights": cfg["training"]["loss_weights"],
        "epoch": epoch,
        "val_metrics": metrics,
    }, path)


def _plot_curves(hist: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("loss"), axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_psnr"]), axes[1].set_title("val PSNR (dB)")
    axes[2].plot(hist["epoch"], hist["val_ssim"]), axes[2].set_title("val SSIM")
    for ax in axes:
        ax.set_xlabel("epoch"), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
