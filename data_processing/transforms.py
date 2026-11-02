import os
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
import yaml
from PIL.Image import Image
from lightly.transforms.multi_crop_transform import MultiCropTranform
from lightly.transforms.multi_view_transform import MultiViewTransform
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import v2 as T

from data_processing.normalization import PerImageNormalize

'''
Available transforms from torchvision:
    - CenterCrop          (size)
    - ColorJitter         (brightness, contrast, saturation, hue := float/(min,max))
    - Grayscale           (num_output_channels=1)
    - Pad                 (padding, fill=0, padding_mode='constant')
    - RandomAffine        (degrees; translate, scale=, shear := tuple; interpolation, fill, fillcolor, resample)
    - RandomApply         (transforms, p=0.5)
    - RandomCrop          (size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
    - RandomGrayscale     (p=0.1)
    - RandomHorizontalFlip(p=0.5)
    - RandomPerspective   (distortion_scale=0.5, p=0.5, interpolation='bilinear', fill=0)
    - RandomResizedCrop   (size, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation='bilinear'>)
    - RandomRotation      (degrees, interpolation='nearest'>, expand=False, center=None, fill=0, resample=None)
    - RandomVerticalFlip  (p=0.5)
    - Resize              (size, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
    - GaussianBlur        (kernel_size, sigma=(0.1, 2.0))
    - RandomChoice        (transforms)
    - RandomOrder         (transforms)
'''


class EvalTransform:
    """Deterministic eval pipeline, plus a training pipeline selected by `mode`.

    mode:
        "same" : train == val == test  (current behaviour, default)
        "ssl"  : train uses BaseTransform, i.e. the same single-view pipeline
                 that TwoViewTransform composes for SSL pretraining
        "mild" : train uses a reduced, supervised-appropriate pipeline
                 (geometry only; no blur, no added noise, gentle crop)

    Exposes .train / .val / .test; __call__ delegates to the eval pipeline so
    existing call sites keep working unchanged.
    """

    MODES = ("same", "ssl", "mild")

    def __init__(
            self,
            normalize: Optional[Dict[str, List[float]]],
            bio_transf: Optional[Module],
            mode: str = "same",
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["transform"]
        resize = cfg["global_size"]

        def _head():
            t = [T.ToImage(),
                 T.ToDtype(torch.float32, scale=False),
                 T.Resize((resize, resize), antialias=True)]
            if bio_transf is not None:
                t += [bio_transf]
            return t

        def _norm():
            if normalize:
                return [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
            return [PerImageNormalize()]

        # ---------------- deterministic eval pipeline ----------------
        self.eval = T.Compose(_head() + _norm())

        # ---------------- training pipeline ----------------
        if mode == "same":
            self.train = self.eval

        elif mode == "ssl":
            # identical to one view of TwoViewTransform
            self.train = BaseTransform(normalize, bio_transf)

        else:  # "mild"
            rrc_scale = tuple(cfg.get("mild_rrc_scale", [0.7, 1.0]))
            ra_degrees = cfg.get("mild_ra_degrees", 15)
            ra_trans = cfg.get("mild_ra_trans", None)
            ra_shear = cfg.get("mild_ra_shear", None)
            gb_prob = cfg.get("mild_gb_prob", 0.0) or 0.0
            gn_prob = cfg.get("mild_gn_prob", 0.0) or 0.0
            gb_sigmas = cfg.get("gb_sigmas")
            gb_kernel = cfg.get("gb_kernel")
            gn_sigmas = cfg.get("gn_sigmas")

            t = _head()
            t += [
                T.RandomResizedCrop(size=resize, scale=rrc_scale, antialias=True),
                T.RandomHorizontalFlip(p=cfg["hf_prob"]),
                T.RandomVerticalFlip(p=cfg["vf_prob"]),
            ]
            if ra_degrees or ra_trans or ra_shear:
                t += [T.RandomAffine(degrees=ra_degrees or 0,
                                     translate=ra_trans, shear=ra_shear)]
            if gb_prob > 0 and gb_kernel is not None:
                t += [T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel,
                                                    sigma=gb_sigmas)], p=gb_prob)]
            t += _norm()
            if gn_prob > 0 and gn_sigmas is not None:
                t += [T.RandomApply([T.GaussianNoise(sigma=gn_sigmas)], p=gn_prob)]
            self.train = T.Compose(t)

        self.val = self.eval
        self.test = self.eval
        self.transform = self.eval          # back-compat with existing attribute use

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.eval(image)

    def get(self) -> Tuple[Any, Any, Any]:
        """Return (train, val, test) transforms."""
        return self.train, self.val, self.test


class TwoViewTransform(MultiViewTransform):
    def __init__(
            self,
            normalize: Optional[Dict[str, Any]] = None,
            bio_transf: Optional[Module] = None,
    ):
        transform_1 = BaseTransform(normalize, bio_transf)
        transform_2 = BaseTransform(normalize, bio_transf)
        self.transforms = [transform_1, transform_2]
        super().__init__(transforms=self.transforms)


class BaseTransform:
    def __init__(
            self,
            normalize: Optional[Dict[str, Any]],
            bio_transf: Optional[Module],
    ):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            config = config["transform"]
        
        global_size = config["global_size"]
        hf_prob = config["hf_prob"]
        vf_prob = config["vf_prob"]
        rrc1, _, rrc2 = config["rrc_scale"]
        rrc_scale = [rrc1, rrc2]
        
        ra_degrees = config["ra_degrees"]
        ra_trans = config["ra_trans"]
        ra_shear = config["ra_shear"]
        
        gb_prob = config["gb_prob"][1]
        gb_sigmas = config["gb_sigmas"]
        gb_kernel = config["gb_kernel"]
        gn_prob = config["gn_prob"]
        gn_sigmas = config["gn_sigmas"]

        transforms = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=False),
            T.Resize((global_size, global_size), antialias=True),
        ]
        
        if bio_transf is not None:
            transforms += [bio_transf]  # arcsinh, logicle, or hyperlog
        
        transforms += [
            T.RandomResizedCrop(size=global_size, scale=rrc_scale),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomAffine(degrees=ra_degrees, translate=ra_trans, shear=ra_shear),
            T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=gb_sigmas)], p=gb_prob),]
        
        # choose normalization: either per dataset or per image (always per channel)
        if normalize:
            transforms += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        else:
            # if no normalization values -> per image normalization
            transforms += [PerImageNormalize()]
        
        # gaussian noise after normalization in standardized units
        transforms += [T.RandomApply([T.GaussianNoise(sigma=gn_sigmas)], p=gn_prob),]
        
        self.transform = T.Compose(transforms)
        

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        transformed: Tensor = self.transform(image)
        return transformed


class DINOTransforms(MultiViewTransform):
    def __init__(
            self,
            normalize: Dict[str, Any] | None,
            bio_transf: Optional[Module],
    ):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            config = config["transform"]

        hf_prob = config["hf_prob"]
        vf_prob = config["vf_prob"]

        rrc_scale = config["rrc_scale"][1:]  #  (0.4, 1.0)
        local_rrc_scale = config["local_rrc_scale"]  # (0.05, 0.4)

        ra_degrees = config["ra_degrees"]  # 0
        ra_trans = config["ra_trans"]  # None
        ra_shear = config["ra_shear"]  # None
        
        gb_prob = config["gb_prob"]  # (1.0, 0.5, 0.1)
        gb_sigmas = config["gb_sigmas"]  # (0.1, 2)
        gb_kernel = config["gb_kernel"]
        gn_prob = config["gn_prob"]  # None
        gn_sigmas = config["gn_sigmas"]  # None

        global_size = config["global_size"]
        local_views = config["local_views"]  # 6
        local_size = config["local_size"]

        # first global crop
        self.global_transform_0 = DINOViewTransform(
            normalize=normalize,
            bio_transf=bio_transf,
            global_size=global_size,
            output_size=global_size,
            rrc_scale=rrc_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            ra_degrees=ra_degrees,
            ra_trans=ra_trans,
            ra_shear=ra_shear,
            gb_prob=gb_prob[0],
            gb_sigmas=gb_sigmas,
            gb_kernel=gb_kernel,
            gn_prob=gn_prob,
            gn_sigmas=gn_sigmas,
        )

        # second global crop
        self.global_transform_1 = DINOViewTransform(
            normalize=normalize,
            bio_transf=bio_transf,
            global_size=global_size,
            output_size=global_size,
            rrc_scale=rrc_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            ra_degrees=ra_degrees,
            ra_trans=ra_trans,
            ra_shear=ra_shear,
            gb_prob=gb_prob[2],
            gb_sigmas=gb_sigmas,
            gb_kernel=gb_kernel,
        )

        # transformation for the local small crops
        self.local_transform = DINOViewTransform(
            normalize=normalize,
            bio_transf=bio_transf,
            global_size=global_size,
            output_size=local_size,
            rrc_scale=local_rrc_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            ra_degrees=ra_degrees,
            ra_trans=ra_trans,
            ra_shear=ra_shear,
            gb_prob=gb_prob[1],
            gb_sigmas=gb_sigmas,
            gb_kernel=gb_kernel,
        )

        local_transforms = [self.local_transform] * local_views
        self.transforms = [self.global_transform_0, self.global_transform_1]
        self.transforms.extend(local_transforms)
        super().__init__(self.transforms)


class DINOViewTransform:
    def __init__(
            self,
            normalize: Dict[str, Any] | None,
            bio_transf: Optional[Module],
            global_size: int = 224,
            output_size: int = 224,
            rrc_scale: Tuple[float, float] = (0.4, 1.0),
            hf_prob: float = 0.5,
            vf_prob: float = 0.5,
            ra_degrees: Optional[float] = None,
            ra_trans: Optional[Tuple[float]] = None,
            ra_shear: Optional[Tuple[float]] = None,
            gb_prob: float = 0,
            gb_sigmas: Optional[float] = None,
            gb_kernel: Optional[int] = None,
            gn_prob: float = 0,
            gn_sigmas: Optional[float] = None,
    ):
        transforms = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=False),
            T.Resize((global_size, global_size), antialias=True),
        ]

        if bio_transf is not None:
            transforms += [bio_transf]  # arcsinh, logicle, or hyperlog
        
        transforms += [
            T.RandomResizedCrop(size=output_size, scale=rrc_scale, interpolation=T.InterpolationMode.BICUBIC,),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomAffine(degrees=ra_degrees, translate=ra_trans, shear=ra_shear),
            T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=gb_sigmas)], p=gb_prob),
        ]

        if normalize:
            transforms += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        else:
            # if no normalization values provided -> per image normalization
            transforms += [PerImageNormalize()]
        
        # gaussian noise after normalization in standardized units
        transforms += [T.RandomApply([T.GaussianNoise(sigma=gn_sigmas)], p=gn_prob),]
        
        self.transform = T.Compose(transforms)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        transformed: Tensor = self.transform(image)
        return transformed