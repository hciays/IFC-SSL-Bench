"""Crop or pad every image listed in a CSV to 48x48, in place.

    python canvas48.py --csv rbc_files_summary.csv --workers 6 --limit 500

Matches Doan et al.: "resized to 48 x 48 pixels by cropping or padding".
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

SIZE = 48
PATH_COLS = ("File Path", "Path", "file_path", "path")
_ROOT = ""
_FILL = None


def to_canvas(img, size, fill):
    """C x H x W -> C x size x size by centre crop / centre pad."""
    c, h, w = img.shape
    if fill is None:                      # per-channel median of the border
        ring = np.concatenate([img[:, 0, :], img[:, -1, :],
                               img[:, :, 0], img[:, :, -1]], axis=1)
        fill = np.median(ring, axis=1)

    out = np.empty((c, size, size), img.dtype)
    out[:] = np.asarray(fill, img.dtype)[:, None, None]

    def win(n):
        if n >= size:
            s = (n - size) // 2
            return slice(s, s + size), slice(0, size)
        d = (size - n) // 2
        return slice(0, n), slice(d, d + n)

    sy, dy = win(h)
    sx, dx = win(w)
    out[:, dy, dx] = img[:, sy, sx]
    return out


def init(root, fill):
    global _ROOT, _FILL
    _ROOT, _FILL = root, fill


def convert(rel):
    import h5py
    try:
        with h5py.File(path, "r+") as f:
            img = f["image"][:]
            print(img.shape)
            2/0
            flip = img.ndim == 3 and img.shape[0] > img.shape[-1]   # H,W,C
            if flip:
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 2:
                img = img[None]

            if img.shape[1:] == (SIZE, SIZE):
                return "skip", rel, ""

            out = to_canvas(img, SIZE, _FILL)
            if flip:
                out = np.transpose(out, (1, 2, 0))

            del f["image"]                 # shape changed, so recreate
            f.create_dataset("image", data=out)
        return "ok", rel, ""
    except Exception as e:
        return "err", rel, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", default="", help="prefix for relative paths in the CSV")
    ap.add_argument("--fill", default=None,
                    help="per-channel pad values, e.g. 791.6,49.4,790.6,72.0 "
                         "(default: median of each image's border)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    col = next((c for c in PATH_COLS if c in df.columns), None)
    if col is None:
        sys.exit(f"no path column in {list(df.columns)}")

    rels = df[col].astype(str).tolist()
    if a.limit:
        rels = rels[: a.limit]
    print(f"{len(rels)} files, path column '{col}'")

    fill = [float(x) for x in a.fill.split(",")] if a.fill else None
    counts = {"ok": 0, "skip": 0, "err": 0}
    errs = []

    with Pool(a.workers, initializer=init, initargs=(a.root, fill)) as pool:
        for i, (status, rel, msg) in enumerate(pool.imap_unordered(convert, rels), 1):
            counts[status] += 1
            if status == "err":
                errs.append((rel, msg))
            if i % 500 == 0 or i == len(rels):
                print(f"\r  {i}/{len(rels)}", end="", flush=True)
    print()
    print(f"{counts['ok']} converted, {counts['skip']} already 48x48, {counts['err']} failed")
    for rel, msg in errs[:10]:
        print(f"  {rel}: {msg}")


if __name__ == "__main__":
    main()