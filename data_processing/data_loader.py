import os
from typing import Tuple, Optional, Dict, Any

import torch
from medmnist import BloodMNIST
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, Subset, random_split

from data_processing.dataset import SplitChannelDataset
from data_processing.dataset_h5 import Dataset_IFC_SSL, DatasetView, H5ImageOnlyManager
from data_processing.normalization import AsinhTransform, LogicleTransform, HyperlogTransform
from data_processing.transforms import TwoViewTransform, EvalTransform, DINOTransforms
from models.model_baselines import DINO
from models.supervised_model import Supervised
from utils.reproducibility import seed_worker
from utils.utils import DATASET_DIR, _subset_split


def _target_to_long(y):
    # y may be np.ndarray([k]) or torch.Tensor([k]) or int
    if isinstance(y, ndarray):
        y = int(y.item()) if y.size == 1 else int(y[0])
    elif isinstance(y, Tensor):
        y = int(y.item())
    else:
        y = int(y)
    return y


def load_from_medmnist(data_info: dict, transforms) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """
    Loads a MedMNIST dataset with train/val/test splits
    """
    train_transform, val_transform, test_transform = transforms
    train_data = BloodMNIST(root=DATASET_DIR, split="train", transform=train_transform, target_transform=_target_to_long,
                            download=True, size=data_info["img_size"])
    val_data = BloodMNIST(root=DATASET_DIR, split="val", transform=val_transform, target_transform=_target_to_long,
                        download=True, size=data_info["img_size"])
    test_data = BloodMNIST(root=DATASET_DIR, split="test", transform=test_transform, target_transform=_target_to_long,
                           download=True, size=data_info["img_size"])

    data_info["classes"] = [v for _, v in train_data.info["label"].items()]
    data_info["train_samples"] = len(train_data)
    data_info["val_samples"] = len(val_data)
    data_info["test_samples"] = len(test_data)

    if data_info["nr_samples"] != -1:  # only part of the dataset
        nr_samples = min(data_info["nr_samples"], len(train_data) + len(val_data) + len(test_data))
        split = (int(0.7 * nr_samples), int(0.15 * nr_samples), int(0.15 * nr_samples))

        train_data = Subset(train_data, range(split[0]))
        val_data = Subset(val_data, range(split[1]))
        test_data = Subset(test_data, range(split[2]))

    print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    return train_data, val_data, test_data, data_info


def load_local_split(data_info: dict, eval: bool, transforms) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """
    Loads a dataset (RBC dataset) with channels split into multiple files per image from a local folder
    """
    
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
    else:
        train_transform, val_transform, test_transform = transforms
        train_path = os.path.join(data_info['data_dir'], "labeled_set", "Training_Canada_Swiss")
        val_path = os.path.join(data_info['data_dir'], "labeled_set", "Test3_HoldOut_Canada_Swiss")
        test_path = os.path.join(data_info['data_dir'], "labeled_set", "Test1_Consensus_groundtruth")
        data_info["classes"] = ["CrenatedDisc", "CrenatedDiscoid", "CrenatedSphere", "CrenatedSpheroid", "Side", "SmoothDisc", "SmoothSphere", "Undecidable"]
    
        train_data = SplitChannelDataset(
            train_path, train_transform, data_info['img_size'], data_info['channels'], data_info['extension'], data_info['classes']
        )
        val_data = SplitChannelDataset(
            val_path, val_transform, data_info['img_size'], data_info['channels'], data_info['extension'], data_info['classes']
        )
        test_data = SplitChannelDataset(
            test_path, test_transform, data_info['img_size'], data_info['channels'], data_info['extension'], data_info['classes']
        )
        data_info["train_samples"] = len(train_data)
        data_info["val_samples"] = len(val_data)
        data_info["test_samples"] = len(test_data)
    
        if data_info["nr_samples"] != -1:  # only part of the dataset
            nr_samples = min(data_info["nr_samples"], len(train_data) + len(val_data) + len(test_data))
            split = (
                round(0.7 * nr_samples / len(data_info["classes"])),
                round(0.15 * nr_samples / len(data_info["classes"])),
                round(0.15 * nr_samples / len(data_info["classes"]))
            )
            print(f"split per class: {split[0]}, {split[1]}, {split[2]} -> {0.7 * nr_samples}, {0.15 * nr_samples}")
    
            train_data = _subset_split(train_path, train_data, split[0], data_info['channels'], data_info['extension'])
            val_data = _subset_split(val_path, val_data, split[1], data_info['channels'], data_info['extension'])
            test_data = _subset_split(test_path, test_data, split[2], data_info['channels'], data_info['extension'])
    
        print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
        return train_data, val_data, test_data, data_info

def load_h5_dataset(data_info: dict, eval: bool, transforms):
    data_path = data_info["data_dir"] + "path_to_labeled_subset" if eval else data_info["data_dir"] + "path_to_unlabeled_subset"
    data_info["classes"] = [
        'B_T_cell_in_one_layer',
        'B_cell',
        #'CD3+ MHCII+ cells',
        #'Dead_Cell',
        'Multiplets',
        'No_cell_cell_interaction',
        'Synapses_with_signaling',
        'Synapses_without_signaling',
        'T_cell',
        'T_cell_with_B_cell_fragments',
        'T_cell_with_signaling',
        #'cropped_images',
        #'no_synapse',
        #'out_of_focus',
        #'unfocused_cells',
        #'unfocused_synapses',
        #'unknown'
    ]

    manager = Dataset_IFC_SSL(
        csv_path=data_info["csv_path"],
        path_override=(".", data_path),
        channels=data_info["channels"],
        experiments=data_info["experiments"],
        donors=data_info["donors"]
    )
    if not eval:
        train_data = manager.get_unlabeled_dataset(transforms)
        data_info["train_samples"] = len(train_data)
        print(f"data size: train: {len(train_data)}")
        return train_data, None, None, data_info
    else:
        train_transform, val_transform, test_transform = transforms
        labeled_data = manager.get_labeled_dataset(data_info["classes"], train_transform)
        # split labeled data into validation and testing
        n_train = int(round(0.64 * len(labeled_data)))
        n_val = int(round(0.16 * len(labeled_data)))

        g = torch.Generator().manual_seed(data_info["seed"])
        perm = torch.randperm(len(labeled_data), generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:(n_train+n_val)]
        test_idx = perm[(n_train+n_val):]

        train_data = DatasetView(labeled_data, train_idx)
        val_data = DatasetView(labeled_data, val_idx)
        test_data = DatasetView(labeled_data, test_idx)
        val_data.base.transforms = val_transform
        test_data.base.transforms = test_transform
        data_info["train_samples"] = len(train_data)
        data_info["val_samples"] = len(val_data)
        data_info["test_samples"] = len(test_data)

        print(f"data size: train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
        return train_data, val_data, test_data, data_info


def set_transform(h_params: dict, data_info: dict, eval: bool):
    """
    image augmentation and normalization setup
    """
    bio_transf = None
    if data_info["dataset"] != "BloodMNIST":
        # fn_intensities only for IFC datasets
        T_top = 65535
        # T set based on the number of bits per pixel
        
        if data_info["bio_transf"] == "asinh":
            bio_transf = AsinhTransform()
        elif data_info["bio_transf"] == "logicle":
            bio_transf = LogicleTransform(T=T_top)
        elif data_info["bio_transf"] == "hyperlog":
            bio_transf = HyperlogTransform(T=T_top)

    if eval or h_params["model"]==Supervised:
        transform = EvalTransform(data_info["norm_vals"], bio_transf).get()
        data_info["transform"] = transform
    else:
        if h_params["model"] not in [DINO, Supervised]:
            transform = TwoViewTransform(data_info["norm_vals"], bio_transf)
            data_info["transform"] = transform.transforms[1].transform
        elif h_params["model"] == DINO:
            transform = DINOTransforms(data_info["norm_vals"], bio_transf)
            data_info["transform"] = [transform.global_transform_0.transform, transform.local_transform.transform]
    return transform, data_info


def load_data(h_params: dict, data_info: dict, eval: bool) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """Call the correct dataset loading function depending on the source."""
    # setup data
    transform, data_info = set_transform(h_params, data_info, eval)
    if h_params["model"] == Supervised:
        eval = True  # supervised model always uses eval transform for training

    if data_info["dataset"] == "BloodMNIST":
        return load_from_medmnist(data_info, transform)
    elif data_info["dataset"] == "RBC_Dataset":
        return load_local_split(data_info, eval, transform)
    elif data_info["dataset"] == "Immuno_Synapses":
        return load_h5_dataset(data_info, eval, transform)
    else:
        raise ValueError(f"load_data: Unsupported dataset: {data_info['dataset']}")


def get_data_loaders(
        train_data: Dataset, val_data: Dataset, test_data: Optional[Dataset], h_params: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps the datasets in PyTorch DataLoaders
    """
    # init data loaders
    try:
        train_loader = DataLoader(
            train_data,
            num_workers=h_params["num_workers"],
            batch_size=8 if len(train_data) < h_params["batch_size"] else h_params["batch_size"],
            persistent_workers=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            pin_memory=True,
            generator=torch.Generator().manual_seed(h_params["seed"])
        )
        # no shuffle on val and test loaders
        val_loader = DataLoader(
            val_data,
            num_workers=h_params["num_workers"],
            batch_size=8 if len(val_data) < h_params["batch_size"] else h_params["batch_size"],
            persistent_workers=True,
            shuffle=False,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_data,
            num_workers=h_params["num_workers"],
            batch_size=8 if len(test_data) < h_params["batch_size"] else h_params["batch_size"],
            persistent_workers=True,
            shuffle=False,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader
    except Exception as e:
        raise Exception(f"Couldn't create data loaders:\n{h_params}\n{e}")


def build_loaders(
        h_params: Dict[str, Any], 
        data_info: Dict[str, Any],
        eval: bool = False,  # marks benchmarking - to differentiate from pretraining
):
    # load_data should respect the EvalTransform based on whether pretrain or eval
    train_data, val_data, test_data, data_info = load_data(h_params, data_info, eval=eval)
    print(f"Train dataset transform: {train_data.transforms}")
    print(f"Val dataset transform: {val_data.transforms}")
    print(f"Test dataset transform: {test_data.transforms}")
    train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data, h_params)

    return {"train": train_loader, "val": val_loader, "test": test_loader}, data_info


def make_weighted_loader(h_params, dataset, class_weights, labels):
    """Create weighted dataloader for supervised model"""
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(h_params["seed"])
    )

    weighted_dataloader = DataLoader(
        dataset,
        batch_size=8 if len(dataset) < h_params["batch_size"] else h_params["batch_size"],
        num_workers=h_params["num_workers"],
        sampler=sampler,
        shuffle=False,  # must be False when sampler is set
        persistent_workers=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
        generator=torch.Generator().manual_seed(h_params["seed"])
    )
    
    return weighted_dataloader
