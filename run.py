import os
from datetime import datetime

import torch
import yaml

from models.backbones import ResNet50, ViTBackbone, ConvNeXtV2, ConvNeXt
from models.model_baselines import SimCLR, MoCoV3, BYOL, BarlowTwins, DINO, MoCoV2
from models.supervised_model import Supervised
from benchmark import benchmark
from pretrain import pretrain
from utils.reproducibility import set_deterministic
from utils.utils import get_checkpoint_dir


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_workers = min(os.cpu_count(), 24)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(model, dataset, bio_transf, scale_norm, test_run, current_epoch, seed):
    # Load config for dataset info
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # setup hyperparameters:
    h_params = {"model": model.__name__}
    h_params.update(config[model.__name__])
    h_params["max_epochs"] = config["max_epochs"]
    if current_epoch > 0: 
        # when continuing a pervious pretraining - calculate number of epochs left
        h_params["max_epochs"] = h_params["max_epochs"] - current_epoch
    h_params["batch_size"] = config["batch_size"]
    h_params["emb_dim"] = config["emb_dim"]

    for key in ("lr", "weight_decay", "momentum"):
        if key in h_params and isinstance(h_params[key], list):
            h_params[key] = [float(elem) for elem in h_params[key]]

    # setup config related to dataset and normalization
    data_info = {
        "dataset": dataset,
        **{k: v for k, v in config[dataset].items() if k != scale_norm},
        "bio_transf": bio_transf,
        "scale_norm": scale_norm,
        "seed": seed,
    }

    # filter normalization lists based on channel list
    norm_vals = config[dataset].get(scale_norm).get(bio_transf)
    channels = config[dataset]["channels"]
    if dataset == "RBC_Dataset":
        # map channels to int
        channel_map = {"Ch1": 0, "Ch9": 1, "Ch12": 2}
        channels = [channel_map[ch] for ch in channels]

    data_info["norm_vals"] = {
        "mean": [norm_vals["mean"][i] for i in channels],
        "std":  [norm_vals["std"][i]  for i in channels],
    }
    
    if "nr_samples" in config:
        # train models on a given number of samples
        data_info["nr_samples"] = config["nr_samples"] 
    else:  
        # train models on all samples
        data_info["nr_samples"] = -1

    # different setup for test-runs - fewer epochs, less samples, etc.
    if test_run and "test" in config:
        if "batch_size" in config["test"]:
            h_params["batch_size"] = config["test"]["batch_size"]
        if "emb_dim" in config["test"]:
            h_params["emb_dim"] = config["test"]["emb_dim"]
        if "max_epochs" in config["test"]:
            h_params["max_epochs"] = config["test"]["max_epochs"]
        if "nr_samples" in config["test"]:
            data_info["nr_samples"] = config["test"]["nr_samples"] 

    return data_info, h_params


def run_pipeline(
        init_type, 
        dataset, 
        backbone, 
        model, 
        bio_transf, 
        scale_norm, 
        test_run, 
        num_workers, 
        seed, 
        fractions,
        run_pretrain=True,
        run_benchmark=True,
        continue_train=False,  # for continuing previous training, from current_epoch up to max_epochs
        current_epoch=0
):
    run_time = datetime.now().strftime("%m.%d-%H'%M")  # for consistent file names

    # Set random seeds for reproducibility
    set_deterministic(seed=seed)

    data_info, h_params = load_config(model, dataset, bio_transf, scale_norm, test_run, current_epoch, seed)
    h_params.update({
        "init_type": init_type,
        "model": model,
        "backbone": backbone,
        "test_run": test_run,
        "num_workers": num_workers,
        "seed": seed,
        "fraction": 0
    })

    # PRETRAIN
    if run_pretrain:
        pretrain(h_params, data_info, run_time, continue_train)
        print(f"pretrain done: {model.__name__}, {backbone.__name__}, {init_type}, {seed}")
    
    # BENCHMARK
    if run_benchmark:
        # check that pretrained model exists
        ckpt_dir = get_checkpoint_dir(h_params, create=False)
        if not ckpt_dir.exists() or len(os.listdir(ckpt_dir)) == 0:
            print(f"\nMISSING: {dataset}, {model.__name__}, {backbone.__name__}, {init_type}, {seed}\n\n")
        else:
            benchmark(h_params, data_info, run_time, fractions)
            print(f"benchmark done: {dataset}, {model.__name__}, {backbone.__name__}, {init_type}, {seed}")


if __name__ == "__main__":
    dataset = os.environ.get("DATASET")
    model = os.environ.get("MODEL")
    backbone = os.environ.get("BACKBONE")
    init_type = os.environ.get("INIT_TYPE")
    seed = int(os.environ.get("SEED"))
    epoch = int(os.environ.get("EPOCH", "0"))

    MODEL_MAP = {
        "Supervised": Supervised,
        "SimCLR": SimCLR,
        "MoCoV3": MoCoV3,
        "BYOL": BYOL,
        "BarlowTwins": BarlowTwins,
        "DINO": DINO,
        "MoCoV2": MoCoV2,
    }
    BACKBONE_MAP = {
        "ResNet50": ResNet50,
        "ViTBackbone": ViTBackbone,
        "ConvNeXtV2": ConvNeXtV2,
        "ConvNeXt": ConvNeXt,
    }
    model = MODEL_MAP[model]
    backbone = BACKBONE_MAP[backbone]
    
    scale_norm = "z-score"
    bio_transf = "none"
    subset_fractions = [
        1, 
        5, 
        10, 
        25, 
        50, 
        100
    ]

    test_run = False

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
                current_epoch=epoch
        )

    except Exception as e:
        print(f"\n\nFailed: {model.__name__}, {backbone.__name__}, {dataset}, {init_type}:\n{e}\n")
        raise e
