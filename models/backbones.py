import math

import timm
import torch
import torch.nn as nn
from timm import create_model
from torch.nn import functional as F



# ======================== CHANNEL ADAPTATION ====================

# The 2x2 ablation for Equation 1. Two independent choices:
#
#                     unscaled        scaled by 3/C
#   mean-replicate    "mean"          "mean_scaled"
#   tile RGB          "tile"          "tile_scaled"   <- timm's adapt_input_conv
#
# "mean" is Equation 1 as reported in the manuscript and is the default.
#
# The scale factor restores the first layer's output magnitude: summing C input
# channels instead of 3 multiplies the pre-activation by roughly C/3, so 3/C
# cancels it. In every backbone here the stem is immediately followed by a
# normalisation layer, which is invariant to a constant rescaling of its input
# (Arora et al., ICLR 2019), so the factor is expected to be absorbed -- except
# in ViT, where the patch embedding enters the residual stream and has the
# positional embedding added to it before any normalisation.
ADAPT_STRATEGIES = ("mean", "mean_scaled", "tile", "tile_scaled")


def adapt_first_conv_weight(w: torch.Tensor, in_chans: int, strategy: str = "mean_scaled") -> torch.Tensor:
    """Map a pretrained RGB first-layer weight to `in_chans` input channels.

    Args:
        w:          pretrained weight, [out_c, 3, kH, kW]
        in_chans:   number of target input channels
        strategy:   one of ADAPT_STRATEGIES

    Returns:
        [out_c, in_chans, kH, kW]
    """
    if strategy not in ADAPT_STRATEGIES:
        raise ValueError(f"strategy must be one of {ADAPT_STRATEGIES}, got {strategy!r}")

    src_chans = w.shape[1]
    w = w.float()

    if strategy.startswith("mean"):
        # Equation 1: average over RGB, replicate to every channel. Symmetric --
        # no channel inherits a spectral identity from ImageNet, which matters
        # because IFC channel order is an instrument setting, not a colour.
        out = w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
    else:
        # timm's adapt_input_conv: tile the RGB filters and truncate. Assigns
        # the red filter to channel 0, green to 1, and so on, cyclically.
        reps = int(math.ceil(in_chans / src_chans))
        out = w.repeat(1, reps, 1, 1)[:, :in_chans, :, :]

    if strategy.endswith("_scaled"):
        out = out * (src_chans / float(in_chans))

    return out.to(w.dtype)


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split("."):
        # nn.Sequential registers children under the string keys "0", "1", ...
        if part.isdigit() and not hasattr(mod, part):
            mod = mod[int(part)]
        else:
            mod = getattr(mod, part)
    return mod

def create_model_adapted(model: str, in_chans: int, num_classes: int = 0,
                         adapt: str = "mean", first_conv: str = None, **kwargs):
    """Create a timm model with `in_chans` channels from ImageNet weights.

    The first conv / patch-embed weights are adapted from the pretrained RGB
    filters according to `adapt`; every deeper layer keeps its pretrained
    weights unchanged.

    `first_conv` overrides the name taken from `default_cfg`, for models
    registered outside timm that do not populate that field.
    """
    base = timm.create_model(model, pretrained=True, in_chans=3,
                             num_classes=num_classes, **kwargs)
    target = timm.create_model(model, pretrained=False, in_chans=in_chans,
                               num_classes=num_classes, **kwargs)

    if first_conv is None:
        first_conv = getattr(target, "default_cfg", {}).get("first_conv", None)
    if first_conv is None:
        raise ValueError(
            f"Cannot adapt input channels for '{model}': no 'first_conv' in "
            f"default_cfg and none supplied. Pass first_conv='<module.path>'."
        )

    base_first = get_submodule(base, first_conv)
    target_first = get_submodule(target, first_conv)

    new_w = adapt_first_conv_weight(base_first.weight.data, in_chans, adapt)
    with torch.no_grad():
        target_first.weight.copy_(new_w)

    # load every other pretrained weight, keeping the adapted first layer
    base_sd = base.state_dict()
    for k in [k for k in base_sd if k.startswith(first_conv + ".weight")]:
        base_sd.pop(k)
    target.load_state_dict(base_sd, strict=False)

    return target


class _TimmBackbone(nn.Module):
    """Shared construction for the timm-registered backbones.

    Subclasses set MODEL_NAME. `adapt` selects the channel-adaptation strategy
    and is only consulted when ImageNet weights are actually loaded for a
    non-RGB input; with random initialisation there is nothing to adapt.
    """

    MODEL_NAME: str = None

    def __init__(self, dataset, channels=None, use_weights: bool = False,
                 adapt: str = "mean", **model_kwargs) -> None:
        super().__init__()
        self.adapt = adapt
        n_ch = len(channels)

        if not use_weights or dataset == "BloodMNIST":
            # no pretrained weights to adapt, or RGB input that needs no adapting
            self.encoder = create_model(
                self.MODEL_NAME,
                pretrained=use_weights,
                in_chans=n_ch,
                num_classes=0,
                global_pool="avg",
                **model_kwargs,
            )
        else:
            self.encoder = create_model_adapted(
                self.MODEL_NAME,
                in_chans=n_ch,
                num_classes=0,
                adapt=adapt,
                global_pool="avg",
                **model_kwargs,
            )
        self.feat_dim = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def describe(self) -> str:
        return f"{self.MODEL_NAME}, adapt={self.adapt}, feat_dim={self.feat_dim}"
    

class ResNet50(_TimmBackbone):
    MODEL_NAME = "resnet50d"


class ConvNeXt(_TimmBackbone):
    MODEL_NAME = "convnext_tiny"


class ConvNeXtV2(_TimmBackbone):
    MODEL_NAME = "convnextv2_tiny"


class ViTBackbone(_TimmBackbone):
    MODEL_NAME = "vit_small_patch16_224"
    PATCH = 16

    def __init__(self, dataset, channels=None, img_size: int = 224,
                 use_weights: bool = False, adapt: str = "mean") -> None:
        self.img_size = ((img_size + self.PATCH - 1) // self.PATCH) * self.PATCH
        super().__init__(dataset, channels, use_weights, adapt,
                         img_size=self.img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if (H != self.img_size) or (W != self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False, antialias=True)
        return self.encoder(x)