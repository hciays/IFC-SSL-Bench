import os
from typing import Callable, Optional, List, Tuple, Sequence

import numpy as np
import torch
from tifffile import imread, TiffFile
from torch import Tensor
from torch.utils.data import Dataset

"""
Dataset Class (like the lightly dataset but adapted to multiple channels)
"""


class SplitChannelDataset(Dataset):  # RBC Dataset
    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable],
            image_size: int,
            channels: list,
            extension: str,
    ) -> None:
        """
        A dataset for multichannel images stored as separate grayscale .ome.tif files per channel.

        Args:
            root_dir: Root path containing class subfolders with .ome.tif images.
            transform: Optional transform applied after stacking channels.
            image_size: Final image size (images are resized to this square dimension).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.channels = channels
        self.extension = extension

        self.labels = []
        self.data = self._build_data()  # Precomputed list of valid samples

    def _build_data(self) -> List[dict]:
        """
        Builds a list of samples with all required channels available.

        Returns: List of dicts with keys:
                - 'paths': list of image paths for the required channels
                - 'label': numeric class label
        """
        data = []

        # Find all class folders under root_dir
        class_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_name)
            files = os.listdir(class_path)

            # Group files by base ID (excluding channel suffix)
            sample_keys = {}
            for f in files:
                if not f.endswith(self.extension):
                    continue
                key = "_".join(f.split("_")[:-1])  # e.g., 'sample1' from 'sample1_Ch1.ome.tif'
                sample_keys.setdefault(key, []).append(f)

            # Only include samples where all required channels are present
            for key, group_files in sample_keys.items():
                if all(f"{key}_{ch}{self.extension}" in group_files for ch in self.channels):
                    data.append({
                        "paths": [os.path.join(class_path, f"{key}_{ch}{self.extension}") for ch in self.channels],
                        "label": label
                    })
                    self.labels.append(label)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        sample = self.data[idx]
        imgs = []

        # Load, convert to grayscale, resize, and collect each channel
        for path in sample["paths"]:
            # preserves original dtype/bit-depth (instead of PIL.Image.open)
            arr = imread(path)
            # Expect single-plane grayscale per file (H,W) - Squeeze extra dims
            arr = np.squeeze(arr)
            imgs.append(arr)

        # Stack channels to shape: [C, H, W]
        # Hand off raw array; transforms handle ToImage/ToDtype/resize
        img = np.stack(imgs, axis=0)

        img = torch.from_numpy(img).to(torch.float32)  # CHW tensor, then cast

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return img, sample["label"]


class SingleChannelDataset(Dataset):  # iPSC Datasets
    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable] = None,
            image_size: int = 256,
            nr_chs: int = 5,
            channels: Optional[Sequence[int]] = None,
            extension: str = ".tiff",
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.nr_chs = nr_chs
        self.extension = extension

        if channels is None:
            self.channels_idx = None
        else:
            # store as sorted list of unique indices
            self.channels_idx = sorted(set(channels))

        self.labels = []
        self.data = self._collect_files()

    def _collect_files(self):
        data = []
        # Find all class folders under root_dir
        class_dirs = sorted(os.listdir(self.root_dir))

        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                if not file.endswith(self.extension):
                    continue
                file_path = os.path.join(class_path, file)
                # check that file is ok / can be opened
                try:
                    with TiffFile(file_path) as tif:
                        _ = tif.pages[0].shape  # Try reading first page metadata
                except Exception as e:
                    continue
                data.append((file_path, label))
                self.labels.append(label)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = imread(path)  # raw dtype

        # Ensure channel-first format [C, H, W]
        if img.ndim == 3 and img.shape[-1] == self.nr_chs:  # nr channels last
            img = np.moveaxis(img, -1, 0)  # [C,H,W]
        elif img.ndim == 3 and img.shape[0] == self.nr_chs:  # nr channels first
            pass  # already [C,H,W]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # select channels from given list
        if self.channels_idx is not None:
            img = img[self.channels_idx, ...]  # [C_sel, H, W]
        
        img = torch.from_numpy(img).to(torch.float32)  # cast to tensor

        if self.transform:
            img = self.transform(img)

        return img, label
