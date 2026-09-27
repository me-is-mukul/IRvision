"""Live terminal view of running training jobs (refreshes every few seconds).

Reads outputs/models/<run>/history.csv, which the trainer rewrites after every epoch.
A run counts as active until it writes summary.json.

Usage:
    python scripts/watch_training.py              # all active runs
    python scripts/watch_training.py r2_unet      # specific runs (also finished ones)
Stop with Ctrl+C.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from irvision.utils.config import load_config

REFRESH_S = 5


def bar(done: int, total: int, width: int = 24) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def render(runs: list[str], models_dir) -> str:
    lines = [f"IRVision training monitor   {time.strftime('%H:%M:%S')}   (Ctrl+C to stop)", ""]
    active = runs or sorted(p.name for p in models_dir.iterdir()
                            if (p / "history.csv").exists() and not (p / "summary.json").exists())
    if not active:
        return "\n".join(lines + ["No active training runs."])
    for run in active:
        hist_path = models_dir / run / "history.csv"
        if not hist_path.exists():
            lines.append(f"{run}: waiting for the first epoch ...")
            continue
        h = pd.read_csv(hist_path)
        done = len(h)
        summary = models_dir / run / "summary.json"
        total = json.loads(summary.read_text())["epochs_run"] if summary.exists() else None
        last = h.iloc[-1]
        best = h.loc[h["val_psnr"].idxmax()]
        eta = ""
        if total is None and "seconds" in h:
            eta = f"   ~{last['seconds']:.0f} s/epoch"
        status = "finished" if summary.exists() else "running"
        lines.append(f"{run}   [{status}]   epoch {done}{'/' + str(total) if total else ''}{eta}")
        if total:
            lines.append(f"  {bar(done, total)}")
        lines.append(f"  latest : loss {last['train_loss']:.4f} | val PSNR {last['val_psnr']:.2f} dB | val SSIM {last['val_ssim']:.4f}")
        lines.append(f"  best   : epoch {int(best['epoch'])} | val PSNR {best['val_psnr']:.2f} dB | val SSIM {best['val_ssim']:.4f}")
        spark = h["val_psnr"].tail(30).tolist()
        lo, hi = min(spark), max(spark)
        ticks = "▁▂▃▄▅▆▇█"
        lines.append("  val PSNR trend: " + "".join(ticks[int((v - lo) / (hi - lo + 1e-9) * 7)] for v in spark))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    runs = sys.argv[1:]
    models_dir = load_config()["paths"]["models_dir"]
    try:
        while True:
            text = render(runs, models_dir)
            print("\033[2J\033[H" + text, flush=True)   # clear screen, cursor home
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
