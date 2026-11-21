import os
from typing import Callable, Optional, List, Tuple, Sequence, Dict

import numpy as np
import torch
from tifffile import imread, TiffFile
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path


"""
Dataset Class (like the lightly dataset but adapted to multiple channels)
"""


def find_class_dirs(root_dir: str, class_file_list: List[str], case_sensitive: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    
    Args:
        root_dir: Root directory to search.
        class_file_list: List of class names/subpaths to match.
        case_sensitive: If False, matches directories regardless of case.
        
    Returns:
        label_to_paths: Dict mapping label name -> sorted list of absolute directory paths.
        label_to_idx: Dict mapping label name -> integer code (0, 1, 2...).
    """
    root = Path(root_dir).resolve()
    
    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()
    
    # Build suffixes to match (strip extension, normalize separators)
    suffixes = []
    for item in class_file_list:
        s = Path(item).stem                 
        # If your list items are just class names 
        # If your list items are relative paths
        # Assuming item is just the class name or a path where we want the last part:
        s_clean = s.replace("\\", "/").rstrip("/")
        suffixes.append(norm(s_clean))
    
    labels_to_paths = {norm(label): [] for label in suffixes}
    label_to_idx = {norm(label): idx for idx, label in enumerate(suffixes)}
    print(f"label_to_idx: {label_to_idx}")
    
    # Iterate through all directories
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        
        p = d.resolve()
        p_str = norm(p.as_posix())
        
        # Check against all requested suffixes
        for suf in suffixes:
            if p_str.endswith(suf):
                # Boundary check: ensure it's a full directory name match
                # e.g. suffix "Cell" should not match "/data/MyCell"
                i = len(p_str) - len(suf)
                if i == 0 or p_str[i - 1] == "/":
                    labels_to_paths[norm(suf)].append(str(p))
                    break # Stop checking other suffixes for this directory once matched
    
    return labels_to_paths, label_to_idx

class SplitChannelDataset(Dataset):  # RBC Dataset
    def __init__(
            self,
            root_dir: str,
            transforms: Optional[Callable],
            image_size: int,
            channels: list,
            extension: str,
            class_list: Optional[List[str]] = None,
    ) -> None:
        """
        A dataset for multichannel images stored as separate grayscale .ome.tif files per channel.

        Args:
            root_dir: Root path containing class subfolders with .ome.tif images.
            transforms: Optional transforms applied after stacking channels.
            image_size: Final image size (images are resized to this square dimension).
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_size = image_size
        self.channels = channels
        self.extension = extension

        self.labels = []
        self.class_list = class_list
        self.data = self._build_data()  # Precomputed list of valid samples

    def _build_data(self) -> List[dict]:
        """Build the list of samples that have every required channel."""
        data = []

        label_to_paths, label_to_idx = find_class_dirs(
            root_dir=self.root_dir,
            class_file_list=self.class_list,
            case_sensitive=False,
        )

        for label, paths in sorted(label_to_paths.items()):
            n_before = len(data)
            for class_path in sorted(paths):
                # group by the full path prefix, i.e. everything before _<Ch>.<ext>,
                # so the key already carries the directory and two samples with the
                # same basename in different subfolders cannot collide
                sample_keys = {}
                for p in sorted(Path(class_path).rglob(f"*{self.extension}")):
                    key = str(p.parent / "_".join(p.name.split("_")[:-1]))
                    sample_keys.setdefault(key, set()).add(p.name)

                for key in sorted(sample_keys):
                    group_files = sample_keys[key]
                    base = os.path.basename(key)
                    if all(f"{base}_{ch}{self.extension}" in group_files for ch in self.channels):
                        data.append({
                            "paths": [f"{key}_{ch}{self.extension}" for ch in self.channels],
                            "label": label_to_idx[label],
                        })
                        self.labels.append(label_to_idx[label])

            print(f"Found for label {label}: {len(paths)} directories, "
                  f"{len(data) - n_before} complete samples")

        if not data:
            raise RuntimeError(
                f"No samples found under {self.root_dir}. Checked for files matching "
                f"*_<ch>{self.extension} with ch in {self.channels}. Verify the "
                f"extension and the channel suffixes against the actual filenames."
            )
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
        # Hand off raw array; let transforms handle ToImage/ToDtype/resize
        img = np.stack(imgs, axis=0)

        img = torch.from_numpy(img).to(torch.float32)  # CHW tensor, then cast

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, sample["label"]