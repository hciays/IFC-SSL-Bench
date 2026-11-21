import os
import csv
import random
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np
from lightning.pytorch.loggers import WandbLogger
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

from data_processing.dataset import SplitChannelDataset


# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

config_path = os.path.join(PROJECT_ROOT, "config.yml")
with open(config_path, "r") as f:
        config = yaml.safe_load(f)

# Base directory for outputs - prefer config path if available, else project root
out_dir = config["out_dir"] if config["out_dir"] != "" else PROJECT_ROOT
OUT_DIR = Path(os.environ.get("OUT_DIR", out_dir)).resolve()
# Logs
LOGS_DIR = OUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Checkpoints - prefer env var path, else same as output dir
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", str(OUT_DIR))).resolve()
CKPT_DIR = CKPT_ROOT / "check_pts"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
# Dataset - prefer env var path, else config if available, else datasets folder in project root
data_dir = config["data_dir"] if config["data_dir"] != "" else PROJECT_ROOT / "datasets"
DATASET_DIR = Path(os.environ.get("DATA_DIR", data_dir)).resolve()


def get_loss_folder(h_params: dict, dataset: str) -> str:
    if h_params["test_run"]:
        folder = Path(f'{LOGS_DIR}/losses/test/{dataset}/{h_params["seed"]}/{h_params["backbone"].__name__}/{h_params["init_type"]}/{h_params["fraction"]}%')
    else:
        folder = Path(f'{LOGS_DIR}/losses/{dataset}/{h_params["seed"]}/{h_params["backbone"].__name__}/{h_params["init_type"]}/{h_params["fraction"]}%')
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def get_loss_path(h_params, data_info, run_time):
    folder = get_loss_folder(h_params, data_info["dataset"])
    file_name = f"{run_time}-{h_params['model'].__name__}.png"
    return os.path.join(folder, file_name)


def get_checkpoint_path(h_params, finetune=False, get_last=False):
    ckpt_dir = get_checkpoint_dir(h_params, finetune)
    if get_last: 
        # get last checkpoint when continuing an existing run
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("last")]
    else:  
        # general case: get best
        files = [f for f in os.listdir(ckpt_dir) if not f.startswith("last")]
    # if multiple files - get last created one
    file_name = sorted(files, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))[-1]
    return f"{ckpt_dir}/{file_name}"


def get_checkpoint_dir(h_params, finetune=False, create=True):
    if not finetune:
        ckpt_base = Path(f"{CKPT_DIR}") 
    else:
        ckpt_base = Path(f"{CKPT_DIR}/finetune")
    
    ckpt_dir = Path(f"{ckpt_base}/{h_params['model'].__name__}/{h_params['backbone'].__name__}/{h_params['init_type']}")
    if not finetune:
        ckpt_dir = Path(f"{ckpt_dir}/{h_params['seed']}")
    else:
        ckpt_dir = Path(f"{ckpt_dir}/{h_params['seed']}/{h_params['fraction']}%")
    
    if h_params['model'].__name__ == "Supervised":
        ckpt_dir = Path(f"{ckpt_dir}/{h_params['fraction']}%")
    
    if create:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_benchmark_dir(data_info, h_params, run_time: str):
    if not h_params['test_run']:
        eval_folder = Path(f"{LOGS_DIR}/benchmark/{data_info['dataset']}/{h_params['seed']}/{h_params['backbone'].__name__}"
                           f"/{h_params['init_type']}/{h_params['fraction']}%")
    else:
        eval_folder = Path(f"{LOGS_DIR}/benchmark/test/{data_info['dataset']}/{h_params['seed']}/{h_params['backbone'].__name__}"
                           f"/{h_params['init_type']}/{h_params['fraction']}%")
    eval_folder.mkdir(parents=True, exist_ok=True)
    return str(eval_folder), f"{eval_folder}/{run_time}"


def setup_logger(h_params: dict, data_info: dict, run_time: str, eval: bool = False) -> WandbLogger:
    """Create a WandB logger"""
    parts = [h_params["model"].__name__,
             h_params["backbone"].__name__,
             data_info["dataset"],
             h_params["init_type"],
             f"s{h_params['seed']}"]
    if eval:
        parts.append(f"f{h_params['fraction']}")
    run_name = "-".join(parts)
 
    # everything except the seed and the fraction: all repetitions of one
    # configuration land in the same group
    group = "-".join([data_info["dataset"], h_params["model"].__name__,
                      h_params["backbone"].__name__, h_params["init_type"]])
 
    config = {
        "dataset": data_info["dataset"],
        "model": h_params["model"].__name__,
        "backbone": h_params["backbone"].__name__,
        "init_type": h_params["init_type"],
        "adapt": h_params.get("adapt", "mean"),
        "seed": h_params["seed"],
        "fraction": h_params.get("fraction", 100),
        "batch_size": h_params["batch_size"],
        "max_epochs": h_params["max_epochs"],
        "emb_dim": h_params["emb_dim"],
        "lr": h_params["lr"],
        "bio_transf": data_info["bio_transf"],
        "scale_norm": data_info["scale_norm"],
        "split_seed": data_info.get("split_seed"),
        "run_time": run_time,
        "transform": str(data_info.get("transform")),
    }
 
    return WandbLogger(
        project=f"{h_params['model'].__name__}-{'eval' if eval else 'train'}",
        name=run_name,
        group=group,
        job_type="benchmark" if eval else "pretrain",
        tags=[data_info["dataset"], h_params["model"].__name__,
              h_params["backbone"].__name__, h_params["init_type"],
              f"seed{h_params['seed']}"],
        config=config,
        checkpoint_name=run_name,
    )


def get_params_text(h_params, data_info):
    params = (
        f"run: {h_params['init_type']}, "
        f"epochs: {h_params['max_epochs']}, "
        f"batch: {h_params['batch_size']}, "
        f"emb: {h_params['emb_dim']}, "
        f"lr: {h_params['lr']}, "
        f"channels:{data_info['channels']}, "
        f"seed: {h_params['seed']}, "
        f"fraction: {h_params['fraction']}, "
    )
    return pformat(params, compact=True, width=80)


def _get_dataset_labels(dataset):
    """Given a dataset, return an array of integer labels"""
    if dataset.__class__.__name__ == "BloodMNIST":
        return np.asarray(dataset.labels.flatten())
 
    if hasattr(dataset, "dataset") and dataset.dataset.__class__.__name__ == "BloodMNIST":
        return np.asarray(dataset.dataset.labels.flatten())
 
    if hasattr(dataset, "df"):
        # a DatasetView keeps its own sliced df but reads label_map from .base
        src = getattr(dataset, "base", dataset)
        if not getattr(src, "is_labeled", False):
            raise ValueError(
                f"{type(dataset).__name__} holds unlabelled data; it has no labels "
                f"to return. _get_dataset_labels was called on the wrong split."
            )
        labels = dataset.df["Label"].astype(str).to_numpy()
        return np.array([src.label_map[s] for s in labels])
 
    return np.asarray(dataset.labels)


def get_class_weights(dataset, subset_indices=None):
    """Calculate and return a list of the class weights for each sample of a dataset"""
    if subset_indices is not None:
        labels = _get_dataset_labels(dataset)[subset_indices]
    else:
        labels = _get_dataset_labels(dataset)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid div0
    return labels, class_weights


def make_subset(dataset, fraction, seed, min_per_class: int = 1):
    """Stratified annotation subset of `dataset`, nested across fractions."""
    if not 0 < fraction <= 100:
        raise ValueError(f"fraction must be in (0, 100], got {fraction}")
 
    labels = np.asarray(_get_dataset_labels(dataset))
 
    # A dedicated Generator, so the chain cannot be perturbed by unrelated draws
    # elsewhere in the process. Classes are visited in sorted order, so the
    # sequence of permutation calls depends only on the seed, never on fraction.
    rng = np.random.default_rng(seed)
 
    chosen = []
    for cls in np.unique(labels):
        pool = np.flatnonzero(labels == cls)
        order = rng.permutation(pool)                     # one ordering per (seed, class)
        k = int(round(fraction / 100.0 * len(pool)))
        k = min(max(k, min_per_class), len(pool))
        chosen.append(order[:k])                          # prefix -> nested
 
    subset_idx = np.sort(np.concatenate(chosen))
    return Subset(dataset, subset_idx), subset_idx

def sample_ids(dataset):
    """Stable, machine-independent identifier per sample, or None."""
    df = getattr(dataset, "df", None)                       # Immuno_Synapses
    if df is not None and "Path" in getattr(df, "columns", []):
        return df["Path"].astype(str).tolist()
 
    data = getattr(dataset, "data", None)                   # RBC
    if isinstance(data, list) and data and isinstance(data[0], dict) and "paths" in data[0]:
        return [str(d["paths"][0]) for d in data]
 
    return None                                             # BloodMNIST, fixed order
 
 
def _relativise(ids, data_info):
    """Strip the machine-specific prefix so a released manifest is portable."""
    root = str(data_info.get("data_dir", "") or "")
    if not root:
        return ids
    return [os.path.relpath(i, root) if os.path.isabs(i) and i.startswith(root) else i
            for i in ids]
 
 
def log_subset(dataset, subset_idx, data_info, h_params, split, class_names=None):
    """Record which labels a run actually consumed. Returns the total."""
    labels = np.asarray(_get_dataset_labels(dataset))
    subset_idx = np.asarray(subset_idx)
    sel = labels[subset_idx]
 
    counts = {}
    for c in np.unique(labels):
        name = class_names[int(c)] if class_names is not None else f"class_{int(c)}"
        counts[name] = int((sel == c).sum())
    total = int(len(sel))
 
    out = os.path.join(get_loss_folder(h_params, dataset), "subsets")
    os.makedirs(out, exist_ok=True)
    stem = f"{data_info['dataset']}_{split}_f{h_params['fraction']}_seed{h_params['seed']}"
 
    idx_path = os.path.join(out, f"{stem}.csv")
    if not os.path.exists(idx_path):
        ids = sample_ids(dataset)
        if ids is not None:
            ids = _relativise(ids, data_info)
        with open(idx_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "label", "sample_id"])
            for i in subset_idx:
                i = int(i)
                w.writerow([i, int(labels[i]), ids[i] if ids is not None else ""])
 
    summary = os.path.join(out, "summary.csv")
    key = (str(data_info["dataset"]), str(split),
           str(h_params["fraction"]), str(h_params["seed"]))
    seen = set()
    if os.path.exists(summary):
        with open(summary, newline="") as f:
            seen = {tuple(r[:4]) for r in csv.reader(f)}
    if key not in seen:
        write_header = not os.path.exists(summary)
        with open(summary, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["dataset", "split", "fraction", "seed", "n_total"] + list(counts))
            w.writerow(list(key) + [total] + [counts[k] for k in counts])
 
    return total

def _count_valid_samples(class_path: str, channels: List[str], extension: str):
    files = os.listdir(class_path)
    sample_keys = {}

    for f in files:
        if not f.endswith(extension):
            continue
        # remove channel and extension from name
        base_img_name = "_".join(f.split("_")[:-1])
        # init img list, then add file to it
        sample_keys.setdefault(base_img_name, []).append(f)

    valid_imgs = [
        img_nm for img_nm, img_chs in sample_keys.items()
        if all(f"{img_nm}_{ch}{extension}" in img_chs for ch in channels)  # if all channels present
    ]
    return valid_imgs  # return the valid keys


def _subset_split(root_dir: str, dataset: SplitChannelDataset, nr_samples: int, channels: List[str], extension: str):
    """
    Returns a Subset(dataset_obj, indices) with balanced sampling from each class.
    """
    # Get class folders
    class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    lookup_idx = {}
    # Build map of sample keys to dataset indices
    for idx, item in enumerate(dataset.data):
        # item contains the paths to all channels
        img_name = os.path.basename(item["paths"][0])  # contains channel name
        base_name = "_".join(img_name.split("_")[:-1])  # base name of image (no channel)
        lookup_idx[base_name] = idx

    # Build class-to-index list
    label = 0
    subset_indices = []
    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        # count number of images that have all channels - avoid those with missing channels
        valid_imgs = _count_valid_samples(class_path, channels, extension)

        if len(valid_imgs) < nr_samples:
            print(f"Not enough samples in class {class_name}. Getting {len(valid_imgs)} instead.")
        else:
            valid_imgs = random.sample(valid_imgs, nr_samples)
        indices = [lookup_idx[k] for k in valid_imgs]
        subset_indices.extend(indices)

        label += 1

    return torch.utils.data.Subset(dataset, subset_indices)


def count_samples_split(root_dir: str, channels: List[str], extension: str):
    """
    Count number of samples for a given dataset.
    Returns a dict of class: nr_samples for each class in the dataset.
    """
    # Get class folders
    class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    # Build class-to-index list
    class_count = {}
    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        # count number of images that have all channels - avoid those with missing channels
        class_count[class_name] = _count_valid_samples(class_path, channels, extension)

    print(f"{root_dir}:\n {class_count}")
    return class_count


def count_samples_single(root_dir: str, extension: str):
    """
    Count number of samples for a given dataset.
    Returns a dict of class: nr_samples for each class in the dataset.
    """
    # Collect class folders
    class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_count = {}
    
    # count nr of samples in each class
    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        imgs = [f for f in os.listdir(class_path) if f.endswith(extension)]
        class_count[class_name] = len(imgs)
    
    print(f"{root_dir}:\n {class_count}")
    return class_count
