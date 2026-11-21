import csv
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from medmnist import BloodMNIST
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, Subset

from data_processing.dataset import SplitChannelDataset
from data_processing.dataset_h5 import Dataset_IFC_SSL, DatasetView, H5ImageOnlyManager
from data_processing.normalization import LogicleTransform
from data_processing.transforms import TwoViewTransform, EvalTransform, DINOTransforms
from models.model_baselines import DINO
from models.supervised_model import Supervised
from utils.reproducibility import seed_worker
from utils.utils import DATASET_DIR, _subset_split, _get_dataset_labels

SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.64, 0.16, 0.20)   # train / val / test


def _target_to_long(y):
    if isinstance(y, ndarray):
        y = int(y.item()) if y.size == 1 else int(y[0])
    elif isinstance(y, Tensor):
        y = int(y.item())
    else:
        y = int(y)
    return y


def _write_split_manifest(data_info, base, train_idx, val_idx, test_idx, labels):
    """Write the train/val/test partition once, so it can be released and checked."""
    out = os.path.join(str(data_info.get("out_dir", ".")), "splits")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{data_info['dataset']}_split_seed{SPLIT_SEED}.csv")
    if os.path.exists(path):
        return

    df = getattr(base, "df", None)
    paths = df["Path"].astype(str).tolist() if df is not None and "Path" in df.columns else None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "split", "label", "sample_id"])
        for name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            for i in idxs:
                i = int(i)
                w.writerow([i, name, labels[i], paths[i] if paths else ""])
    print(f"[splits] wrote {path}")


def load_from_medmnist(data_info: dict, transforms) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """Loads a MedMNIST dataset with its published train/val/test splits."""
    train_transform, val_transform, test_transform = transforms
    train_data = BloodMNIST(root=DATASET_DIR, split="train", transform=train_transform,
                            target_transform=_target_to_long, download=True, size=data_info["img_size"])
    val_data = BloodMNIST(root=DATASET_DIR, split="val", transform=val_transform,
                          target_transform=_target_to_long, download=True, size=data_info["img_size"])
    test_data = BloodMNIST(root=DATASET_DIR, split="test", transform=test_transform,
                           target_transform=_target_to_long, download=True, size=data_info["img_size"])

    data_info["classes"] = [v for _, v in train_data.info["label"].items()]
    data_info["train_samples"] = len(train_data)
    data_info["val_samples"] = len(val_data)
    data_info["test_samples"] = len(test_data)

    if data_info["nr_samples"] != -1:
        nr_samples = min(data_info["nr_samples"], len(train_data) + len(val_data) + len(test_data))
        split = (int(0.7 * nr_samples), int(0.15 * nr_samples), int(0.15 * nr_samples))
        train_data = Subset(train_data, range(split[0]))
        val_data = Subset(val_data, range(split[1]))
        test_data = Subset(test_data, range(split[2]))

    print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    return train_data, val_data, test_data, data_info


def load_local_split(data_info: dict, eval: bool, transforms) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """RBC: channels split across files, partition as released by the original authors."""
    if not eval:
        manager = H5ImageOnlyManager(
                csv_path=os.path.join(data_info['data_dir'], "unlabeled", "rbc_files_summary.csv"),
                image_channels=data_info['channels-unlabeled'],
                path_override=(".", data_info['data_dir']),
            )
        train_data = manager.get_dataset(transforms)
        data_info["train_samples"] = len(train_data)
        print(f"data size: train: {len(train_data)}")
        return train_data, None, None, data_info

    train_transform, val_transform, test_transform = transforms
    train_path = os.path.join(data_info['data_dir'], "labeled_set", "Training_Canada_Swiss")
    val_path = os.path.join(data_info['data_dir'], "labeled_set", "Test3_HoldOut_Canada_Swiss")
    test_path = os.path.join(data_info['data_dir'], "labeled_set", "Test1_Consensus_groundtruth")
    data_info["classes"] = ["CrenatedDisc", "CrenatedDiscoid", "CrenatedSphere", "CrenatedSpheroid",
                            "Side", "SmoothDisc", "SmoothSphere", "Undecidable"]

    train_data = SplitChannelDataset(train_path, train_transform, data_info['img_size'],
                                     data_info['channels'], data_info['extension'], data_info['classes'])
    val_data = SplitChannelDataset(val_path, val_transform, data_info['img_size'],
                                   data_info['channels'], data_info['extension'], data_info['classes'])
    test_data = SplitChannelDataset(test_path, test_transform, data_info['img_size'],
                                    data_info['channels'], data_info['extension'], data_info['classes'])
    data_info["train_samples"] = len(train_data)
    data_info["val_samples"] = len(val_data)
    data_info["test_samples"] = len(test_data)

    if data_info["nr_samples"] != -1:
        nr_samples = min(data_info["nr_samples"], len(train_data) + len(val_data) + len(test_data))
        n_cls = len(data_info["classes"])
        split = (round(0.7 * nr_samples / n_cls),
                 round(0.15 * nr_samples / n_cls),
                 round(0.15 * nr_samples / n_cls))
        print(f"split per class: {split[0]}, {split[1]}, {split[2]}")
        train_data = _subset_split(train_path, train_data, split[0], data_info['channels'], data_info['extension'])
        val_data = _subset_split(val_path, val_data, split[1], data_info['channels'], data_info['extension'])
        test_data = _subset_split(test_path, test_data, split[2], data_info['channels'], data_info['extension'])

    print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    return train_data, val_data, test_data, data_info


def load_h5_dataset(data_info: dict, eval: bool, transforms):
    """Immuno_Synapses: one labelled pool, partitioned here with a fixed seed."""
    data_path = data_info["data_dir"] + "/testing" if eval else data_info["data_dir"] + "/from_natcomm/training"
    data_info["classes"] = [
        'B_T_cell_in_one_layer',
        'B_cell',
        'Multiplets',
        'No_cell_cell_interaction',
        'Synapses_with_signaling',
        'Synapses_without_signaling',
        'T_cell',
        'T_cell_with_B_cell_fragments',
        'T_cell_with_signaling',
    ]

    manager = Dataset_IFC_SSL(
        csv_path=data_info["csv_path"],
        path_override=(".", data_path),
        channels=data_info["channels"],
        experiments=data_info["experiments"],
        donors=data_info["donors"],
    )

    if not eval:
        train_data = manager.get_unlabeled_dataset(transforms)
        data_info["train_samples"] = len(train_data)
        print(f"data size: train: {len(train_data)}")
        return train_data, None, None, data_info

    train_transform, val_transform, test_transform = transforms

    labeled_data = manager.get_labeled_dataset(data_info["classes"], train_transform)

    n = len(labeled_data)

    labels = np.asarray(_get_dataset_labels(labeled_data))
    frac_train, frac_val, _ = SPLIT_FRACTIONS

    train_idx, rest_idx = train_test_split(
        np.arange(n), train_size=frac_train, random_state=SPLIT_SEED,
        stratify=labels, shuffle=True)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=frac_val / (1.0 - frac_train), random_state=SPLIT_SEED,
        stratify=labels[rest_idx], shuffle=True)
    
    train_data = DatasetView(labeled_data, train_idx.tolist())
    val_data = DatasetView(labeled_data, val_idx.tolist())
    test_data = DatasetView(labeled_data, test_idx.tolist())
    val_data.base.transforms = val_transform
    test_data.base.transforms = test_transform
    

    data_info["train_samples"] = len(train_data)
    data_info["val_samples"] = len(val_data)
    data_info["test_samples"] = len(test_data)
    data_info["split_seed"] = SPLIT_SEED

    _write_split_manifest(data_info, labeled_data, train_idx, val_idx, test_idx, labels)

    print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}"
          f"   (fixed partition, split_seed={SPLIT_SEED})")
    return train_data, val_data, test_data, data_info


def set_transform(h_params: dict, data_info: dict, eval: bool):
    """Image augmentation and normalisation setup."""
    bio_transf = None
    if data_info["dataset"] != "BloodMNIST":
        T_top = 65535
        if data_info["bio_transf"] == "logicle":
            bio_transf = LogicleTransform(T=T_top)

    if eval or h_params["model"] == Supervised:
        transform = EvalTransform(data_info["norm_vals"], bio_transf).get()
        data_info["transform"] = transform
    else:
        if h_params["model"] not in [DINO, Supervised]:
            transform = TwoViewTransform(data_info["norm_vals"], bio_transf)
            data_info["transform"] = transform.transforms[1].transform
        elif h_params["model"] == DINO:
            transform = DINOTransforms(data_info["norm_vals"], bio_transf)
            data_info["transform"] = [transform.global_transform_0.transform,
                                      transform.local_transform.transform]
    return transform, data_info


def load_data(h_params: dict, data_info: dict, eval: bool) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """Dispatch to the loader for this dataset."""
    transform, data_info = set_transform(h_params, data_info, eval)
    if h_params["model"] == Supervised:
        eval = True  # the supervised arm always works on the labelled split

    if data_info["dataset"] == "BloodMNIST":
        return load_from_medmnist(data_info, transform)
    elif data_info["dataset"] == "RBC_Dataset":
        return load_local_split(data_info, eval, transform)
    elif data_info["dataset"] == "Immuno_Synapses":
        return load_h5_dataset(data_info, eval, transform)
    raise ValueError(f"load_data: unsupported dataset: {data_info['dataset']}")


def _one_loader(data, h_params, shuffle: bool) -> Optional[DataLoader]:
    """None in, None out: pretraining has no validation or test split."""
    if data is None:
        return None
    kwargs = dict(
        num_workers=h_params["num_workers"],
        batch_size=8 if len(data) < h_params["batch_size"] else h_params["batch_size"],
        persistent_workers=h_params["num_workers"] > 0,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(h_params["seed"])
    return DataLoader(data, **kwargs)


def get_data_loaders(train_data, val_data, test_data, h_params):
    """Wrap the datasets in DataLoaders; val and test are not shuffled."""
    try:
        return (_one_loader(train_data, h_params, shuffle=True),
                _one_loader(val_data, h_params, shuffle=False),
                _one_loader(test_data, h_params, shuffle=False))
    except Exception as e:
        raise Exception(f"Couldn't create data loaders:\n{h_params}\n{e}")


def build_loaders(
        h_params: Dict[str, Any],
        data_info: Dict[str, Any],
        eval: bool = False,  # marks benchmarking, as opposed to pretraining
):
    train_data, val_data, test_data, data_info = load_data(h_params, data_info, eval=eval)

    for name, d in (("Train", train_data), ("Val", val_data), ("Test", test_data)):
        # val and test are None during pretraining
        if d is not None:
            print(f"{name} dataset transform: {getattr(d, 'transforms', None)}")

    train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data, h_params)
    return {"train": train_loader, "val": val_loader, "test": test_loader}, data_info


def make_weighted_loader(h_params, dataset, class_weights, labels):
    """Class-balanced sampler for the supervised model."""
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(h_params["seed"]),
    )

    return DataLoader(
        dataset,
        batch_size=8 if len(dataset) < h_params["batch_size"] else h_params["batch_size"],
        num_workers=h_params["num_workers"],
        sampler=sampler,
        shuffle=False,  # must be False when a sampler is set
        persistent_workers=h_params["num_workers"] > 0,
        worker_init_fn=seed_worker,
        pin_memory=True,
        generator=torch.Generator().manual_seed(h_params["seed"]),
    )