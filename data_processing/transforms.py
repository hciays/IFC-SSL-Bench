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
    """Preprocessing and augmentation for every supervised-objective stage.

    One pipeline, no variants. The same augmentation is applied to the
    supervised baselines, fine-tuning and linear probing, so that differences
    between methods reflect the learned representation rather than the
    augmentation budget.

        train : resize -> bio-transform -> normalise -> gentle geometry
        val   : resize -> bio-transform -> normalise
        test  : resize -> bio-transform -> normalise

    Augmentation is geometry only: a gentle crop, horizontal and vertical
    flips, and rotation up to 15 degrees. Cells in flow have no canonical
    orientation, so all four are label-preserving. There is no blur and no
    added noise; self-supervised pretraining uses the stronger view-generating
    pipeline of each objective, since contrastive objectives require stronger
    augmentation than supervised training (SimCLR, Sec. 3.2).

    Parameters are class constants rather than config entries, so the recipe
    reported in the manuscript and the recipe that runs are a single source of
    truth that can be checked in one place.

    Ordering
    --------
    Geometry is applied AFTER normalisation. Rotation fills revealed corners
    with 0, and normalisation is what makes 0 mean "channel background": raw 0
    is far outside the data range of a brightfield channel (RBC brightfield
    background is ~791 with std ~76, so a raw-0 fill would sit roughly ten
    standard deviations below the mean and appear as a black wedge). Operating
    on normalised tensors keeps the fill at the channel mean.

    Exposes .train / .val / .test; __call__ delegates to the deterministic
    pipeline so existing call sites keep working unchanged.
    """

    RRC_SCALE = (0.7, 1.0)
    HFLIP_P = 0.5
    VFLIP_P = 0.5
    ROTATION_DEG = 15.0

    IMG_SIZE = 224

    def __init__(
            self,
            normalize: Optional[Dict[str, List[float]]],
            bio_transf: Optional[Module],
            img_size: int = IMG_SIZE,
    ):
        self.img_size = int(img_size)

        # deterministic: to float tensor, resample to a fixed grid, bio-transform
        head = [T.ToImage(),
                T.ToDtype(torch.float32, scale=False),
                T.Resize((self.img_size, self.img_size), antialias=True)]
        if bio_transf is not None:
            head += [bio_transf]

        if normalize:
            norm = [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        else:
            norm = [PerImageNormalize()]

        # stochastic, label-preserving, applied to normalised tensors
        geometry = [
            T.RandomResizedCrop(size=self.img_size, scale=self.RRC_SCALE, antialias=True),
            T.RandomHorizontalFlip(p=self.HFLIP_P),
            T.RandomVerticalFlip(p=self.VFLIP_P),
            T.RandomRotation(degrees=self.ROTATION_DEG),
        ]

        self.eval = T.Compose(head + norm)
        self.train = T.Compose(head + norm + geometry)
        self.val = self.eval
        self.test = self.eval
        self.transform = self.eval          # back-compat with existing attribute use

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.eval(image)

    def get(self) -> Tuple[Any, Any, Any]:
        """Return (train, val, test) transforms."""
        return self.train, self.val, self.test

    def describe(self) -> str:
        """One-line summary of the training pipeline, for run logs."""
        return (f"{self.img_size}px, normalise then geometry: "
                f"RandomResizedCrop{self.RRC_SCALE}, "
                f"HFlip(p={self.HFLIP_P}), VFlip(p={self.VFLIP_P}), "
                f"Rotation(±{self.ROTATION_DEG:g}°)")


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
    """One SSL view.

    Order: resample -> bio-transform -> normalise -> geometry -> blur -> noise.

    Normalisation runs BEFORE the geometric transforms. RandomAffine fills the
    corners it reveals with 0, and normalisation is what makes 0 mean "channel
    background". On raw intensities it does not: brightfield background is
    bright, so a raw-0 fill sits far below anything real -- about -10 sigma on
    RBC brightfield and -23 sigma on ISF channel 0. With degrees=180 and
    translate=0.15 roughly 18% of each view is fill, and the filled fraction
    tracks the rotation angle, which is a shortcut an invariance objective can
    exploit. Operating on normalised tensors puts the fill at the channel mean.

    Normalisation and blur commute (a unit-sum kernel passes through an affine
    rescaling unchanged), so moving it earlier does not change the blur.
    Gaussian noise stays last, with sigma in standardised units.
    """

    def __init__(
            self,
            normalize: Optional[Dict[str, Any]],
            bio_transf: Optional[Module],
    ):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["transform"]

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
            transforms += [bio_transf]          # logicle

        # --- normalise first, so that 0 == channel background ---------------
        if normalize:
            transforms += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        else:
            transforms += [PerImageNormalize()]

        # --- geometry on normalised tensors ---------------------------------
        transforms += [
            T.RandomResizedCrop(size=global_size, scale=rrc_scale, antialias=True),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomAffine(degrees=ra_degrees, translate=ra_trans, shear=ra_shear),
            T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=gb_sigmas)], p=gb_prob),
        ]

        # gaussian noise last, sigma in standardised units
        transforms += [T.RandomApply([T.GaussianNoise(sigma=gn_sigmas)], p=gn_prob)]

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
    """One DINO view, global or local.

    Same ordering as BaseTransform and for the same reason: normalise before
    geometry so RandomAffine's zero fill lands at the channel mean.

    Interpolation is bilinear with antialiasing, matching every other view
    pipeline. It was previously bicubic here, which made the resampling kernel
    co-vary with the objective and confounded DINO with the other methods.
    """

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
            transforms += [bio_transf]

        # --- normalise first, so that 0 == channel background ---------------
        if normalize:
            transforms += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        else:
            transforms += [PerImageNormalize()]

        # --- geometry on normalised tensors ---------------------------------
        transforms += [
            T.RandomResizedCrop(size=output_size, scale=rrc_scale, antialias=True),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomAffine(degrees=ra_degrees, translate=ra_trans, shear=ra_shear),
            T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=gb_sigmas)], p=gb_prob),
        ]

        # gaussian noise last, sigma in standardised units
        transforms += [T.RandomApply([T.GaussianNoise(sigma=gn_sigmas)], p=gn_prob)]

        self.transform = T.Compose(transforms)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        transformed: Tensor = self.transform(image)
        return transformed