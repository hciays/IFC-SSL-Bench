import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_loss_folder

SERIES_COLOUR = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"


# --------------------------------------------------------------------------- #
#  write side: call this from pretrain.py
# --------------------------------------------------------------------------- #

def save_loss_curve(train_loss_epoch, h_params, data_info):
    """Append one run's per-epoch losses as a tidy CSV row set."""
    out_root = get_loss_folder(h_params, data_info['dataset'])
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"{data_info['dataset']}.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "model", "backbone", "init_type", "seed", "epoch", "loss"])
        for epoch, loss in enumerate(train_loss_epoch, start=1):
            w.writerow([data_info["dataset"], h_params["model"].__name__,
                        h_params["backbone"].__name__, h_params["init_type"],
                        h_params["seed"], epoch, f"{loss:.6f}"])
    return path


# --------------------------------------------------------------------------- #
#  plot side
# --------------------------------------------------------------------------- #

def _read(path, dataset=None, backbone=None, init_type=None):
    runs = defaultdict(lambda: defaultdict(list))   # model -> seed -> [(epoch, loss)]
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if dataset and r["dataset"] != dataset:
                continue
            if backbone and r["backbone"] != backbone:
                continue
            if init_type and r["init_type"] != init_type:
                continue
            runs[r["model"]][r["seed"]].append((int(r["epoch"]), float(r["loss"])))
    return runs


def _aggregate(per_seed):
    """-> epochs, mean, lo, hi  across seeds, truncated to the shortest run."""
    series = []
    for seed in sorted(per_seed):
        pts = sorted(per_seed[seed])
        series.append(np.array([v for _, v in pts]))
    n = min(len(s) for s in series)
    stack = np.vstack([s[:n] for s in series])
    return np.arange(1, n + 1), stack.mean(0), stack.min(0), stack.max(0), len(series)


def plot(runs, out_path, title=None, ncols=3):
    models = sorted(runs)
    n = len(models)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    # ~3.3 in per panel: two panels fit a single column, three a double column
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows),
                             squeeze=False, constrained_layout=True)

    for k, model in enumerate(models):
        ax = axes[k // ncols][k % ncols]
        epochs, mean, lo, hi, n_seeds = _aggregate(runs[model])

        ax.fill_between(epochs, lo, hi, color=SERIES_COLOUR, alpha=0.20, linewidth=0)
        ax.plot(epochs, mean, color=SERIES_COLOUR, linewidth=1.6, solid_capstyle="round")

        # the panel title identifies the series, so identity is never colour alone
        ax.set_title(f"{model}  (n = {n_seeds})", fontsize=8, color=INK,
                     loc="left", pad=4)
        ax.set_xlim(1, epochs[-1])
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=3, width=0.6)
        if k // ncols == nrows - 1:
            ax.set_xlabel("Epoch", fontsize=8, color=INK_MUTED)
        if k % ncols == 0:
            ax.set_ylabel("Training loss", fontsize=8, color=INK_MUTED)

    for k in range(n, nrows * ncols):             # blank any unused cell
        axes[k // ncols][k % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")

    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--init-type", default=None)
    ap.add_argument("--out", default="loss_curves.pdf")
    a = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,     # embed TrueType, so the text stays editable
        "ps.fonttype": 42,
    })

    path = os.path.join(a.csv_dir, f"{a.dataset}.csv")
    runs = _read(path, a.dataset, a.backbone, a.init_type)
    if not runs:
        raise SystemExit(f"no runs matched in {path}")
    subtitle = " · ".join(x for x in (a.dataset, a.backbone, a.init_type) if x)
    plot(runs, a.out, title=subtitle)


if __name__ == "__main__":
    main()