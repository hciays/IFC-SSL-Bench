import os

import gc
import torch

from models.backbones import ResNet50, ConvNeXt, ConvNeXtV2, ViTBackbone
from models.model_baselines import SimCLR, MoCoV2, MoCoV3, BYOL, BarlowTwins, DINO
from models.supervised_model import Supervised
from run import run_pipeline


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_workers = min(24, os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_combos():
    datasets = [
        "BloodMNIST", 
        "RBC_Dataset", 
        "Immuno_Synapses"
    ]
    models   = [
        Supervised, 
        SimCLR, 
        MoCoV2, 
        MoCoV3, 
        BYOL, 
        BarlowTwins, 
        DINO
    ]
    backbones = [
        ResNet50, 
        ConvNeXtV2, 
        ViTBackbone, 
        ConvNeXt
    ]    
    init_types = [
        "imagenet_weights",
        "no_weights"
    ]
    seeds = [
        7,
        2025,
        42,
    ]
    bio_transfs = [
        "none",
        "asinh",
        "logicle",
        "hyperlog"
    ]

    rows = []
    
    for seed in seeds:
        for bio_transf in bio_transfs:
            for init_type in init_types:
                for dataset in datasets:
                    for backbone in backbones:
                        for model in models:
                            rows.append((dataset, model, backbone, init_type, bio_transf, seed))

    return rows


if __name__ == "__main__":
    subset_fractions = [
        1, 
        5, 
        10, 
        25, 
        50, 
        100
    ]
    scale_norm = "z-score"
    test_run = False

    combos = make_combos()

    for row in combos:
        dataset, model, backbone, init_type, bio_transf, seed = row

        if dataset == "Immuno_Synapses" and model.__name__ == "Supervised":
            continue
        try:
            run_pipeline(
                    init_type, 
                    dataset, 
                    backbone, 
                    model, 
                    bio_transf, 
                    scale_norm, 
                    test_run, 
                    num_workers, 
                    seed, 
                    subset_fractions,
            )
        except torch.OutOfMemoryError as e:
            print(f"[OOM] Skipping run {row}: {e}")
            # free GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            continue
