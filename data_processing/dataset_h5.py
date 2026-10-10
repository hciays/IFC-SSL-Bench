import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
import os
from typing import List, Tuple, Optional, Dict, Any
 

class DatasetView(Dataset):
    """A view of a base dataset with a fixed list of indices, forwarding attributes."""
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)

        # convenient: keep a split-specific df if base has one
        if hasattr(base_dataset, "df"):
            self.df = base_dataset.df.iloc[self.indices].reset_index(drop=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

    def __getattr__(self, name):
        # forward everything else (label_map, num_classes, channels, etc.)
        return getattr(self.base, name)
    

class IFC_H5_Dataset(Dataset):
    """
    A PyTorch Dataset class for handling H5 IFC image data with CSV metadata.
    from https://www.nature.com/articles/s41467-023-43429-2
    
    This class reads the CSV directly and applies all filters (State, Label,
    Experiment, Donor, Antibodies) during initialization.
    """
 
    def __init__(
        self,
        csv_path: str,
        channels: List[int],
        path_override_map: Optional[Tuple[str, str]],
        transforms: Optional[Any] = None,
        is_labeled: bool = False,
        label_map: Optional[Dict[str, int]] = None,
        num_classes: Optional[int] = None,
        # --- Filters ---
        filter_state: Optional[str] = None,
        filter_labels: Optional[List[str]] = None,
        filter_experiments: Optional[List[str]] = None,
        filter_donors: Optional[List[str]] = None,
        filter_antibodies: Optional[List[str]] = None
    ):
        """
        Args:
            csv_path (str): Path to the CSV file.
            channels (List[int]): List of channel indices to extract.
            path_override_map (Tuple[str, str], optional): (old_root, new_root).
            transforms (callable, optional): Transforms to apply.
            is_labeled (bool): Whether to return labels and metadata.
            label_map (Dict[str, int], optional): Mapping from label string to integer index.
            num_classes (int, optional): Total number of classes for One-Hot encoding.
            filter_state (str, optional): Filter for 'State' (e.g., 'labeled').
            filter_labels (List[str], optional): Filter for 'Label'.
            filter_experiments (List[str], optional): Filter for 'Experiment'.
            filter_donors (List[str], optional): Filter for 'Donor'.
            filter_antibodies (List[str], optional): Filter for 'Antibodies'.
        """
        self.csv_path = csv_path
        self.channels = channels
        self.path_override_map = path_override_map
        self.transforms = transforms
        self.is_labeled = is_labeled
        self.label_map = label_map
        self.num_classes = num_classes
 
        # --- Read CSV ---
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # --- Apply Filters ---
        # 1. State (labeled/unlabeled)
        if filter_state:
            df = df[df['State'] == filter_state]
            
        # 2. Labels (specific cell types)
        if filter_labels:
            df = df[df['Label'].isin(filter_labels)]
            
        # 3. Experiments (Global Filter)
        if filter_experiments:
            df = df[df['Experiment'].isin(filter_experiments)]
            
        # 4. Donors (Global Filter)
        if filter_donors:
            df = df[df['Donor'].isin(filter_donors)]
            
        # 5. Antibodies (Global Filter)
        # print(df["Antibody_or_Condition"].unique())
        if filter_antibodies:
            df = df[df['Antibodies'].isin(filter_antibodies)]
            
        # Reset index
        self.df = df.reset_index(drop=True)
        
        if len(self.df) == 0:
            print(f"Warning: Dataset initialized with 0 samples.")
            print(f"Filters applied: State={filter_state}, Labels={filter_labels}, "
                  f"Exp={filter_experiments}, Donor={filter_donors}, Antibodies={filter_antibodies}")
 
    def __len__(self) -> int:
        return len(self.df)
 
    def _get_path(self, original_path: str) -> str:
        """Applies path override logic."""
        if self.path_override_map:
            old, new = self.path_override_map
            if original_path.startswith(old):
                return original_path.replace(old, new, 1)
        return original_path
 
    def _load_and_process_image(self, full_path: str) -> Optional[torch.Tensor]:
        """Opens H5, handles channel ordering, selects channels, and converts to Tensor."""
        try:
            with h5py.File(full_path, 'r') as f:
                if 'image' not in f:
                    raise KeyError(f"Key 'image' not found in {full_path}")
                
                raw_data = f['image'][:]
 
                # --- Heuristic for Channel First vs Last ---
                shape = raw_data.shape
                if shape[0] < shape[-1]:
                    # (C, H, W) -> Already Channel First
                    image_data = raw_data
                else:
                    # (H, W, C) -> Channel Last -> Transpose to (C, H, W)
                    image_data = np.transpose(raw_data, (2, 0, 1))
 
                # --- Channel Selection ---
                selected_channels = [image_data[i] for i in self.channels]
                image = np.stack(selected_channels, axis=0)
                
                # Ensure float32
                image = image.astype(np.float32)
                return torch.from_numpy(image)
 
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return None
 
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 1. Construct Path
        # full_path = self._get_path(os.path.join(row['Path'], row['Filename']))
        full_path = self._get_path(row["Path"])
        
        # 2. Load Image
        image = self._load_and_process_image(full_path)
        
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
 
        # 3. Apply Transforms (Supports SimCLR tuple return)
        if self.transforms:
            image = self.transforms(image)
 
        # 4. Return Data
        if self.is_labeled:
            experiment = row['Experiment']
            donor = row['Donor']
            antibody = row['Antibody_or_Condition']
            raw_label = row['Label']
            
            # One-Hot Encoding
            label_idx = self.label_map[raw_label]
            # one_hot_label = torch.zeros(self.num_classes)
            # one_hot_label[label_idx] = 1.0
            
            return {
                "image": image,
                "experiment": experiment,
                "donor": donor,
                "antibody": antibody,
                "label": label_idx,  # one_hot_label,
                "label_name": raw_label
            }
        else:
            return {
                "image": image
            }
 
 
class Dataset_IFC_SSL:
    """
    A Manager/Factory class.
    
    Allows filtering by Experiment, Donor, or Antibodies at the initialization level.
    These filters persist for all datasets (labeled or unlabeled) created by this manager.
    """
 
    def __init__(
        self,
        csv_path: str,
        path_override: Optional[Tuple[str, str]] = None,
        channels: List[int] = [0, 1, 2],
        # --- Global Filters ---
        experiments: Optional[List[str]] = None,
        donors: Optional[List[str]] = None,
        antibodies: Optional[List[str]] = None
    ):
        """
        Args:
            csv_path (str): Path to the CSV file.
            path_override (Tuple[str, str], optional): Tuple containing (old_root, new_root).
            channels (List[int]): List of channel indices to use.
            experiments (List[str], optional): List of Experiment IDs to include (e.g. ['Exp1', 'Exp2']).
            donors (List[str], optional): List of Donor IDs to include (e.g. ['DonorA']).
            antibodies (List[str], optional): List of Antibodies to include.
        """
        self.csv_path = csv_path
        self.path_override = path_override
        self.channels = channels
        
        # Store global filters
        self.experiments = experiments
        self.donors = donors
        self.antibodies = antibodies
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
 
    def get_labeled_dataset(self, target_labels: List[str], transforms: Optional[Any] = None) -> IFC_H5_Dataset:
        """
        Creates a dataset for Fine-tuning (Labeled data), applying global filters + label filters.
        """
        # Create Label Mapping (String -> Index)
        unique_labels = sorted(list(set(target_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        
        print(f"Creating Labeled Dataset...")
        print(f"Global Filters -> Exp: {self.experiments}, Donor: {self.donors}, Antibodies: {self.antibodies}")
        print(f"Target Labels: {unique_labels}")
        
        return IFC_H5_Dataset(
            csv_path=self.csv_path,
            channels=self.channels,
            path_override_map=self.path_override,
            transforms=transforms,
            is_labeled=True,
            label_map=label_map,
            num_classes=len(unique_labels),
            filter_state='labeled',
            filter_labels=target_labels,
            # Pass global filters
            filter_experiments=self.experiments,
            filter_donors=self.donors,
            filter_antibodies=self.antibodies
        )
 
    def get_unlabeled_dataset(self, transforms: Optional[Any] = None) -> IFC_H5_Dataset:
        """
        Creates a dataset for Pretraining (Unlabeled data), applying global filters.
        """
        print(f"Creating Unlabeled Dataset...")
        print(f"Global Filters -> Exp: {self.experiments}, Donor: {self.donors}, Antibodies: {self.antibodies}")
        
        return IFC_H5_Dataset(
            csv_path=self.csv_path,
            channels=self.channels,
            path_override_map=self.path_override,
            transforms=transforms,
            is_labeled=False,
            filter_state='unlabeled',
            filter_labels=None,
            # Pass global filters
            filter_experiments=self.experiments,
            filter_donors=self.donors,
            filter_antibodies=self.antibodies
        )