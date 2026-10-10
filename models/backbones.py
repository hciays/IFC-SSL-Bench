import timm
import torch
import torch.nn as nn
from timm import create_model
from torch.nn import functional as F


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def create_model_avg_in_chans(model: str, in_chans: int, num_classes: int = 0, **kwargs):
    """
    Create a timm model `model` with `in_chans` input channels, where the first conv/patch-embed weights are
    initialized from the RGB mean and repeated across channels. All deeper layers use pretrained ImageNet weights.
    """
    # Base 3-channel pretrained model
    base = timm.create_model(
        model,
        pretrained=True,
        in_chans=3,
        num_classes=num_classes,
        **kwargs,
    )

    # Target model with desired input channels
    target = timm.create_model(
        model,
        pretrained=False,
        in_chans=in_chans,
        num_classes=num_classes,
        **kwargs,
    )

    first_conv_name = target.default_cfg.get("first_conv", None)
    base_first = get_submodule(base, first_conv_name)
    target_first = get_submodule(target, first_conv_name)
    w = base_first.weight.data

    # Average over RGB and repeat to in_chans
    mean_rgb = w.mean(dim=1, keepdim=True)              # [out_c, 1, kH, kW]
    avg_weight = mean_rgb.repeat(1, in_chans, 1, 1)     # [out_c, in_chans, kH, kW]

    with torch.no_grad():
        target_first.weight.copy_(avg_weight)

    # Load all other pretrained weights
    base_sd = base.state_dict()
    # Drop weights for the first conv (already set)
    drop_prefixes = [first_conv_name + ".weight", first_conv_name + ".bias"]
    for k in list(base_sd.keys()):
        if any(k.startswith(p) for p in drop_prefixes):
            base_sd.pop(k)
    # Load remaining weights non-strictly (so the new first layer is kept)
    target.load_state_dict(base_sd, strict=False)

    return target


# ======================== BACKBONES ====================


class ResNet50(nn.Module):
    def __init__(self, dataset, channels = None, use_weights: bool = False) -> None:
        super().__init__()
        if not use_weights or dataset == "BloodMNIST":
            # create model with no weights, or with weights and rgb input
            self.encoder = create_model(
                "resnet50d",
                pretrained=use_weights,     # loads ImageNet weights when True
                in_chans=len(channels),     # replaces first conv/patch embed for custom channels
                num_classes=0,              # no classifier layer; forward() returns features
                global_pool="avg",          # GAP -> [B, D]
            )
        else:
            # create model with weights and variable number of input channels.
            # use average of rgb imagenet weights for first layer
            self.encoder = create_model_avg_in_chans(
                "resnet50d",
                in_chans=len(channels),
                num_classes=0,
                global_pool="avg",
            )
        self.feat_dim = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # With num_classes=0, global_pool='avg' -> returns [B, D]
        return self.encoder(x)

    def freeze(self) -> None:
        # Disable gradient computation for all layers
        for param in self.parameters():
            param.requires_grad = False
        self.encoder.eval()


class ConvNeXt(nn.Module):
    def __init__(self, dataset, channels = None, use_weights: bool = False) -> None:
        super().__init__()
        if not use_weights or dataset == "BloodMNIST":
            self.encoder = create_model(
                "convnext_tiny",
                pretrained=use_weights,  # loads ImageNet weights when True
                in_chans=len(channels),  # replaces first conv/patch embed for custom channels
                num_classes=0,           # no classifier layer; forward() returns features
                global_pool="avg",       # GAP -> [B, D]
            )
        else:
            self.encoder = create_model_avg_in_chans(
                "convnext_tiny",
                in_chans=len(channels),
                num_classes=0,
                global_pool="avg",
            )
        self.feat_dim = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # With num_classes=0, global_pool='avg' -> returns [B, D]
        return self.encoder(x)

    def freeze(self) -> None:
        # Disable gradient computation for all layers
        for param in self.parameters():
            param.requires_grad = False
        self.encoder.eval()


class ConvNeXtV2(nn.Module):
    def __init__(self, dataset, channels = None, use_weights: bool = False) -> None:
        super().__init__()
        if not use_weights or dataset == "BloodMNIST":
            self.encoder = create_model(
                "convnextv2_tiny",
                pretrained=use_weights,     # loads ImageNet weights when True
                in_chans=len(channels),     # replaces first conv/patch embed for custom channels
                num_classes=0,              # no classifier layer; forward() returns features
                global_pool="avg",          # GAP -> [B, D]
            )
        else:
            self.encoder = create_model_avg_in_chans(
                "convnextv2_tiny",
                in_chans=len(channels),
                num_classes=0,
                global_pool="avg",
            )
        self.feat_dim = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # With num_classes=0, global_pool='avg' -> returns [B, D]
        return self.encoder(x)

    def freeze(self) -> None:
        # Disable gradient computation for all layers
        for param in self.parameters():
            param.requires_grad = False
        self.encoder.eval()


class ViTBackbone(nn.Module):
    def __init__(self, dataset, channels = None, img_size: int = 64, use_weights: bool = False) -> None:
        super().__init__()
        # image size is divisible by patch size (16)
        self.img_size = ((img_size + 16 - 1) // 16) * 16
        if not use_weights or dataset == "BloodMNIST":
            self.encoder = create_model(
                "vit_small_patch16_224",
                pretrained=use_weights,     # loads ImageNet weights when True
                in_chans=len(channels),     # replaces first conv/patch embed for custom channels
                img_size=self.img_size,     # position embeddings will be resized as needed
                num_classes=0,              # no classifier layer; forward() returns features
                global_pool="avg",          # GAP -> [B, D]  (or "token")
            )
        else:
            self.encoder = create_model_avg_in_chans(
                "vit_small_patch16_224",
                in_chans=len(channels),
                num_classes=0,
                global_pool="avg",
                img_size=self.img_size,
            )
        self.feat_dim = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if (H != self.img_size) or (W != self.img_size):
            # keep tensors float32; antialias helps for down/up-sampling
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False, antialias=True)
        return self.encoder(x)

    def freeze(self) -> None:
        # Disable gradient computation for all layers
        for param in self.parameters():
            param.requires_grad = False
        self.encoder.eval()