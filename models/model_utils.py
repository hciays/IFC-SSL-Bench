from typing import Dict, Any

import torch
from lightly.utils.lars import LARS
from lightning import LightningModule
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from models.backbones import ViTBackbone
from models.finetune_model import FinetuneModel
from models.supervised_model import Supervised
from utils.utils import get_checkpoint_path


def build_param_groups(backbone, heads, lr_backbone, lr_head, wd):
    """
    Builds parameter groups based on backbone vs. head, weight decay vs. no decay
    """
    EXCLUDE_WD = ("bias", "bn", "norm", "pos_embed", "cls_token", "relative_position_bias_table", "gamma")
    
    # parameter groups such as bias or batch norm should not be affected by weight decay
    def split(named_params):
        decay, no_decay = [], []
        for name, param in named_params:
            if not param.requires_grad: 
                continue
            if param.ndim < 2 or any(k in name.lower() for k in EXCLUDE_WD):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    decay_backbone, no_decay_backbone = split(backbone.named_parameters())
    named_heads = []
    for h in heads: 
        named_heads += list(h.named_parameters())
    decay_head, no_decay_head = split(named_heads)

    groups = [
        {"params": decay_backbone, "lr": lr_backbone, "weight_decay": wd},
        {"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": decay_head, "lr": lr_head, "weight_decay": wd},
        {"params": no_decay_head, "lr": lr_head, "weight_decay": 0.0},
    ]
    return groups


def build_optimizer(method: str, uses_bn: bool, params, lr: float, weight_decay: float):
    """
    Build optimizer based on model and backbone.
    uses_bn: True for ResNet (BatchNorm), False for ConvNeXt / ViT (LayerNorm).
    """
    if uses_bn:
        # BN backbones
        if method in {"simclr", "byol", "barlowtwins"}:
            # Large-batch SSL methods commonly use LARS with small WD
            return LARS(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif method in {"mocov3", "dino", "mocov2"}:
            # MoCo v3 / DINO with ResNet backbones use SGD with WD 1e-4
            return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            print(f"BUILD OPTIMIZER: none found for {method}")
            return None
    else:
        # ViT / ConvNeXt (LayerNorm) -> AdamW with stronger WD
        return AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def build_scheduler(trainer, optimizer, warmup_ratio: float = 0.05, min_warmup_steps: int = 1):
    """
    Per-step Linear warmup -> Cosine decay
    """
    total_steps = trainer.estimated_stepping_batches
    warmup_steps = max(min_warmup_steps, int(total_steps * warmup_ratio))

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0),
        ],
        milestones=[warmup_steps],
    )
    return {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
    }


@torch.no_grad()
def update_bn_buffers(src: nn.Module, dst: nn.Module):
    # BN buffer sync for BN backbones (prevents teacher BN drift)
    src_bns = [m for m in src.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    dst_bns = [m for m in dst.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    # assume models are isomorphic: zip in traversal order
    for s, d in zip(src_bns, dst_bns):
        d.running_mean.copy_(s.running_mean)
        d.running_var.copy_(s.running_var)
        if hasattr(s, "num_batches_tracked") and hasattr(d, "num_batches_tracked"):
            d.num_batches_tracked.copy_(s.num_batches_tracked)


def create_model(h_params: dict, data_info: dict, class_weights, train: bool) -> LightningModule:
    """
    Builds a model with specified backbone and configuration.
    Returns a LightningModule instance.
    """
    # option to use imagenet-supervised pretrained backbones
    imagenet_pretrained = True if train and h_params["init_type"] == "imagenet_weights" else False
    # create backbone
    if h_params["backbone"] != ViTBackbone:
        backbone = h_params["backbone"](data_info["dataset"], data_info["channels"], imagenet_pretrained)
    else:
        backbone = ViTBackbone(data_info["dataset"], data_info["channels"], data_info["img_size"], imagenet_pretrained)
    
    # create model
    if h_params["model"] != Supervised:
        model = h_params["model"](backbone, h_params)
    else:
        model = Supervised(backbone, h_params, class_weights)
    return model


def load_trained_model(h_params: Dict[str, Any], data_info: Dict[str, Any], class_weights, finetune=False, last_ckpt=False):
    # initializes the correct backbone and model per h_params
    model = create_model(h_params, data_info, class_weights, train=False)
    # get checkpoint path: last checkpoint (e.g., when continuing pretraining), else choose best 
    ckpt_path = get_checkpoint_path(h_params, finetune=finetune, get_last=last_ckpt)

    # load trained model
    if finetune: 
        trained = FinetuneModel.load_from_checkpoint(
            ckpt_path, 
            backbone=model.backbone, 
            n_classes=len(data_info["classes"]), 
            cls_weights=class_weights
        )
    elif h_params["model"] == Supervised:
        # load supervised model after training
        trained = Supervised.load_from_checkpoint(
            ckpt_path, 
            backbone=model.backbone, 
            h_params=h_params, 
            cls_weights=class_weights
        )
    else:
        # load ssl model after pretraining
        trained = h_params["model"].load_from_checkpoint(ckpt_path, backbone=model.backbone, h_params=h_params)
    
    return trained, ckpt_path


def extract_embeddings(model, data_loader, device):
    model.eval().to(device)
    features, labels, experiments = [], [], []

    with torch.no_grad():
        for sample in data_loader:
            if isinstance(sample, dict) and "image" in sample.keys():
                img = sample["image"]
                if "label" in sample.keys():
                    label = sample["label"]
                else: 
                    label = None
                if "experiment" in sample.keys():
                    experiment = sample["experiment"]
                else: 
                    experiment = None

            else:
                img, label = sample
                experiment = None
            
            img = img.to(device)  # ensure data is on same device as model
            feats = model.get_embeddings(img)
            features.append(feats.cpu())
            if label is not None:
                labels.append(label.cpu())
            if experiment is not None:
                experiments.append(experiment)

    features = torch.cat(features).numpy()
    if len(labels) > 0:
        labels = torch.cat(labels).numpy()
    else: 
        labels = None
    if len(experiments) > 0:
        experiments = [e for sub in experiments for e in sub]
    else: 
        experiments = None
    return features, labels, experiments