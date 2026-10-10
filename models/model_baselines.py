import copy
import math

import torch
from lightly.loss import (
    NegativeCosineSimilarity, NTXentLoss, BarlowTwinsLoss, DINOLoss
)
from lightly.models import utils
from lightly.models.modules import heads
from lightning import LightningModule
from torch import nn, distributed
from torch.nn import functional as F, SyncBatchNorm

from models.model_utils import build_optimizer, build_scheduler, update_bn_buffers, build_param_groups


class SimCLR(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        # batchnorm backbones (resnet) work better with a different optimizer than layernorm backbones (convnext, vit)
        # different optimizers need different learning rates
        self._uses_bn = self.uses_batchnorm(backbone)
        
        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.temperature = h_params["temperature"]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        feature_dim = self.backbone.feat_dim
        # Projection head maps backbone features to embedding space
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, self.emb_dim)

        # Contrastive loss function with temperature scaling
        self.loss_function = NTXentLoss(temperature=self.temperature)

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        emb = self.projection_head(feats)
        return F.normalize(emb, dim=1)
    
    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]
        emb0 = self.forward(imgs[0])
        emb1 = self.forward(imgs[1])

        loss = self.loss_function(emb0, emb1)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, _):
        loss = self._calculate_loss(batch, mode="train")
        # step-wise loss
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone, 
            heads=[self.projection_head], 
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay, 
        )
        
        optim = build_optimizer("simclr", self._uses_bn, params, self.lr, self.weight_decay)
        scheduler = build_scheduler(self.trainer, optim)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats


class MoCoV2(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        self._uses_bn = self.uses_batchnorm(backbone)

        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.temperature = h_params["temperature"]
        self.momentum: float = h_params["momentum"][0] if self._uses_bn else h_params["momentum"][1]
        self.memory_bank = h_params["memory_bank"]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        feature_dim = self.backbone.feat_dim
        self.projection_head = heads.MoCoProjectionHead(feature_dim, feature_dim, self.emb_dim)

        # Momentum encoder for target embeddings (EMA)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)
        self.backbone_momentum.eval()
        self.projection_head_momentum.eval()

        # Contrastive loss with memory bank
        self.loss_function = NTXentLoss(temperature=self.temperature, memory_bank_size=(self.memory_bank, self.emb_dim))

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        preds = self.projection_head(feats)
        embs = F.normalize(preds, dim=1)
        return embs

    def forward_momentum(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone_momentum(batch)
        preds = self.projection_head_momentum(feats)
        embs = F.normalize(preds, dim=1).detach()
        return embs

    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())

    def _contrastive_step(self, img_q: torch.Tensor, img_k: torch.Tensor):
        # Batch shuffle required for BatchNorm statistics
        if self.uses_batchnorm(self.backbone):  # BN-only shuffle
            img_k, shuffle = utils.batch_shuffle(img_k)

        # Online encoder
        emb_q = self.forward(img_q)
        # Momentum encoder
        emb_k = self.forward_momentum(img_k)

        if self.uses_batchnorm(self.backbone):
            emb_k = utils.batch_unshuffle(emb_k, shuffle)

        loss = self.loss_function(emb_q, emb_k)
        return loss

    def ema_momentum(self, cur_step: int, max_steps: int, base_m=0.99, end_m=1.0):
        # increases from base_m to end_m
        cos = (1 + math.cos(math.pi * cur_step / max_steps)) / 2
        return end_m - (end_m - base_m) * cos

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        # per-step EMA momentum
        momentum = self.ema_momentum(
            cur_step=self.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            base_m=self.momentum,
            end_m=1.0
        )
        # Update momentum encoder
        utils.update_momentum(self.backbone, self.backbone_momentum, momentum)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, momentum)

        # BN buffer sync for BN backbones (prevents teacher BN drift)
        if self._uses_bn:
            update_bn_buffers(self.backbone, self.backbone_momentum)
            update_bn_buffers(self.projection_head, self.projection_head_momentum)

        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]

        # Symmetric loss: compare both (x0, x1) and (x1, x0)
        loss_1 = self._contrastive_step(imgs[0], imgs[1])
        loss_2 = self._contrastive_step(imgs[1], imgs[0])
        loss = 0.5 * (loss_1 + loss_2)

        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone,
            heads=[self.projection_head],
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay,
        )
        optim = build_optimizer("mocov2", self._uses_bn, params, self.lr, self.weight_decay)
        sched = build_scheduler(self.trainer, optim)  # warmup -> cosine on steps
        return {"optimizer": optim, "lr_scheduler": sched}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats


class MoCoV3(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        self._uses_bn = self.uses_batchnorm(backbone)

        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.temperature = h_params["temperature"]
        self.momentum: float = h_params["momentum"][0] if self._uses_bn else h_params["momentum"][1]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        feature_dim = self.backbone.feat_dim
        self.projection_head = heads.MoCoProjectionHead(feature_dim, feature_dim, self.emb_dim)

        # Momentum encoder for target embeddings (EMA)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)
        self.backbone_momentum.eval()
        self.projection_head_momentum.eval()

        # predictor: 2-layer MLP on ONLINE branch (momentum branch has no predictor)
        self.predictor = self._make_predictor(self.emb_dim)
        if self._uses_bn:
            self.predictor = SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        # Contrastive loss - v3 doesn’t use a memory bank
        self.loss_function = NTXentLoss(temperature=self.temperature)

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
    
    def _make_predictor(self, dim):
        if self._uses_bn:
            pred = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim, bias=True),
            )
            return nn.SyncBatchNorm.convert_sync_batchnorm(pred)
        else:
            return nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim, bias=True),
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)  # raw backbone features
        proj = self.projection_head(feats)
        preds = self.predictor(proj) 
        embs = F.normalize(preds, dim=1)
        return embs

    def forward_momentum(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone_momentum(batch)
            proj = self.projection_head_momentum(feats)
            embs = F.normalize(proj, dim=1).detach()  # stop-grad target
        return embs

    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())
    
    def _contrastive_step(self, img_q: torch.Tensor, img_k: torch.Tensor) -> torch.Tensor:
        # Batch norm backbones benefit from batch shuffle. Skip for ConvNeXt and ViT (layer norm)
        if self._uses_bn:
            img_k, shuffle = utils.batch_shuffle(img_k)

        q = self.forward(img_q)
        k = self.forward_momentum(img_k)

        if self._uses_bn:
            k = utils.batch_unshuffle(k, shuffle)

        loss = self.loss_function(q, k)  # in-batch negatives (no queue)
        return loss
    
    def ema_momentum(self, cur_step: int, max_steps: int, base_m=0.99, end_m=1.0):
        # increases from base_m to end_m
        cos = (1 + math.cos(math.pi * cur_step / max_steps)) / 2
        return end_m - (end_m - base_m) * cos

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        # per-step EMA momentum
        momentum = self.ema_momentum(
            cur_step=self.global_step, 
            max_steps=self.trainer.estimated_stepping_batches,
            base_m=self.momentum, 
            end_m=1.0
        )
        # Update momentum encoder
        utils.update_momentum(self.backbone, self.backbone_momentum, momentum)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, momentum)

        # BN buffer sync for BN backbones (prevents teacher BN drift)
        if self._uses_bn:
            update_bn_buffers(self.backbone, self.backbone_momentum)
            update_bn_buffers(self.projection_head, self.projection_head_momentum)

        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]

        # Symmetric loss: compare both (x0, x1) and (x1, x0)
        loss_1 = self._contrastive_step(imgs[0], imgs[1])
        loss_2 = self._contrastive_step(imgs[1], imgs[0])
        loss = 0.5 * (loss_1 + loss_2)

        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone, 
            heads=[self.projection_head, self.predictor],
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay, 
        )
        
        optim = build_optimizer("mocov3", self._uses_bn, params, self.lr, self.weight_decay)
        scheduler = build_scheduler(self.trainer, optim)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats


class BYOL(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        self._uses_bn = self.uses_batchnorm(backbone)

        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.momentum: float = h_params["momentum"][0] if self._uses_bn else h_params["momentum"][1]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        feature_dim = self.backbone.feat_dim
        # Online network
        self.projection_head = heads.BYOLProjectionHead(feature_dim, feature_dim, self.emb_dim)
        self.prediction_head = heads.BYOLProjectionHead(self.emb_dim, feature_dim, self.emb_dim)

        # Momentum network
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)
        self.backbone_momentum.eval()
        self.projection_head_momentum.eval()

        self.loss_function = NegativeCosineSimilarity()

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(batch)
        proj = self.projection_head(feat)
        pred = self.prediction_head(proj)
        emb = F.normalize(pred, dim=1)
        return emb

    def forward_momentum(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone_momentum(batch)
            proj_feat = self.projection_head_momentum(feat)
            embs = F.normalize(proj_feat, dim=1).detach()  # stop gradient
        return embs

    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())
    
    def ema_momentum(self, cur_step: int, max_steps: int, base_m=0.99, end_m=1.0):
        # cosine increase from base_m -> end_m
        cos = (1 + math.cos(math.pi * cur_step / max_steps)) / 2
        return end_m - (end_m - base_m) * cos

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        # per-step EMA momentum
        momentum = self.ema_momentum(
            cur_step=self.global_step, 
            max_steps=self.trainer.estimated_stepping_batches, 
            base_m=self.momentum, 
            end_m=1.0
        )
        # Update momentum models with EMA
        utils.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        # BN buffer sync for BN backbones
        if self._uses_bn:
            update_bn_buffers(self.backbone, self.backbone_momentum)
            update_bn_buffers(self.projection_head, self.projection_head_momentum)

        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]

        # Online predictions
        p0 = self.forward(imgs[0])
        p1 = self.forward(imgs[1])

        # Target projections
        z0 = self.forward_momentum(imgs[0])
        z1 = self.forward_momentum(imgs[1])

        # Symmetric BYOL loss
        loss = 0.5 * (self.loss_function(p0, z1) + self.loss_function(p1, z0))
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone, 
            heads=[self.projection_head, self.prediction_head],
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay, 
        )
        
        optim = build_optimizer("byol", self._uses_bn, params, self.lr, self.weight_decay)
        scheduler = build_scheduler(self.trainer, optim)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats


class BarlowTwins(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        self._uses_bn = self.uses_batchnorm(backbone)

        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        # 2-layer projection head
        feat_dim = backbone.feat_dim
        self.projection_head = heads.BarlowTwinsProjectionHead(feat_dim, feat_dim, self.emb_dim)
        
        if self._uses_bn and distributed.is_initialized():
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            self.projection_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head)

        self.loss_function = BarlowTwinsLoss(gather_distributed=True)

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def forward(self, batch):
        feats = self.backbone(batch)
        emb = self.projection_head(feats)
        return emb
    
    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]
        emb0 = self.forward(imgs[0])
        emb1 = self.forward(imgs[1])

        loss = self.loss_function(emb0, emb1)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, _):
        loss = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone, 
            heads=[self.projection_head],
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay, 
        )
        
        optim = build_optimizer("barlowtwins", self._uses_bn, params, self.lr, self.weight_decay)
        scheduler = build_scheduler(self.trainer, optim)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats


class DINO(LightningModule):
    def __init__(
            self,
            backbone,
            h_params,
            freeze=False
    ):
        super().__init__()
        self._uses_bn = self.uses_batchnorm(backbone)

        self.max_epochs: int = h_params["max_epochs"]
        self.emb_dim: int = h_params["emb_dim"]
        self.lr: float = h_params["lr"][0] if self._uses_bn else h_params["lr"][1]
        self.lr_head: float = h_params["lr_head"][0] if self._uses_bn else h_params["lr_head"][1]
        self.bottleneck: int = h_params["bottleneck"]
        self.momentum: float = h_params["momentum"][0] if self._uses_bn else h_params["momentum"][1]
        self.weight_decay: float = h_params["weight_decay"][0] if self._uses_bn else h_params["weight_decay"][1]

        self.backbone = backbone
        feature_dim = self.backbone.feat_dim
        self.projection_head = heads.DINOProjectionHead(
            feature_dim, feature_dim, self.bottleneck, self.emb_dim, batch_norm=self._uses_bn
        )

        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, feature_dim, self.bottleneck, self.emb_dim, batch_norm=self._uses_bn
        )
        self.teacher_head.load_state_dict(self.projection_head.state_dict())
        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)
        self.teacher_backbone.eval()
        self.teacher_head.eval()

        if self._uses_bn and distributed.is_initialized():
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            self.projection_head = nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head)

        self.loss_function = DINOLoss(output_dim=self.emb_dim)

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def forward(self, batch):
        feats = self.backbone(batch)
        emb = self.projection_head(feats)
        return emb

    def forward_teacher(self, batch):
        with torch.no_grad():
            feat = self.teacher_backbone(batch)
            emb = self.teacher_head(feat)
        return emb
    
    def uses_batchnorm(self, module: nn.Module) -> bool:
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())
    
    def ema_momentum(self, cur_step: int, max_steps: int, base_m=0.99, end_m=1.0):
        # increases from base_m to end_m
        cos = (1 + math.cos(math.pi * cur_step / max_steps)) / 2
        return end_m - (end_m - base_m) * cos

    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        # step-based momentum
        momentum = self.ema_momentum(
            self.global_step,
            self.trainer.estimated_stepping_batches,
            base_m=self.momentum,
            end_m=1.0,
        )
        
        utils.update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        utils.update_momentum(self.projection_head, self.teacher_head, m=momentum)

        # for BN backbones/heads, also copy BN buffers so teacher stats don’t drift
        if self._uses_bn or self.uses_batchnorm(self.projection_head):
            update_bn_buffers(self.backbone, self.teacher_backbone)
            update_bn_buffers(self.projection_head, self.teacher_head)

        if not isinstance(batch, dict):
            imgs, _ = batch
        else:
            imgs = batch["image"]

        teacher_emb = [self.forward_teacher(v) for v in imgs[:2]]  # first two as globals
        student_emb = [self.forward(v) for v in imgs]   # all crops

        loss = self.loss_function(teacher_emb, student_emb, epoch=self.current_epoch)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._epoch_loss_count += 1

        return loss
    
    def on_train_epoch_end(self):
        if self._epoch_loss_count > 0:
            mean_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
        
        # reset fields
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def configure_optimizers(self):
        # build param groups (decay/no-decay)
        params = build_param_groups(
            backbone=self.backbone, 
            heads=[self.projection_head],
            lr_backbone=self.lr,
            lr_head=self.lr_head,
            wd=self.weight_decay, 
        )
        
        optim = build_optimizer("dino", self._uses_bn, params, self.lr, self.weight_decay)
        scheduler = build_scheduler(self.trainer, optim)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats