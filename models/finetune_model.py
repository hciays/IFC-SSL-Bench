import lightning as L
import torch

from torch import nn, Tensor, set_grad_enabled
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class FinetuneModel(L.LightningModule):
    def __init__(
            self, 
            backbone: nn.Module, 
            n_classes: int, 
            cls_weights,
            freeze: bool = False,
            lr_backbone=1e-4, 
            lr_head=1e-3, 
            weight_decay=1e-4,
    ):
        super().__init__()
        # trined model backbone
        self.backbone = backbone
        feat_dim = self.backbone.feat_dim
        # attach untrained head for classification
        self.head = nn.Linear(feat_dim, n_classes)
        self.loss_function = nn.CrossEntropyLoss(weight=Tensor(cls_weights))
        
        self.freeze: bool = freeze
        self.lr_backbone = lr_backbone  # backbone with lower lr than head
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Global accuracy over all samples, standard overall top-1 accuracy
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        # f1 per class, then average
        self.f1_score = MulticlassF1Score(num_classes=n_classes, average="macro")

        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._step_count = 0

    def forward(self, batch):
        # if fine-tune then don't freeze weights
        with set_grad_enabled(not self.freeze):
            feats = self.backbone(batch)
        return self.head(feats)

    def _step(self, batch, mode):
        if not isinstance(batch, dict):  # small datasets
            imgs, labels = batch
        else:                            # unlabeled dataset
            imgs = batch["image"]
            labels = batch["label"]
        emb = self.forward(imgs)
        
        # calculate loss and metrics
        loss = self.loss_function(emb, labels)
        accuracy = self.accuracy(emb, labels)
        f1_score = self.f1_score(emb, labels)

        # log loss and metrics
        if mode == "ft_val":
            self.log(f"{mode}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"{mode}_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"{mode}_f1m", f1_score, prog_bar=False, on_epoch=True, on_step=False)
    
        return loss

    def training_step(self, batch, _):
        loss = self._step(batch, "ft_train")
        self.train_losses.append(loss.item())
        # epoch aggregation
        self._epoch_loss_sum += float(loss.item())
        self._step_count += 1

        return loss
    
    def validation_step(self, batch, _): 
        return self._step(batch, "ft_val")
    
    def test_step(self, batch, _): 
        return self._step(batch, "ft_test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if not isinstance(batch, dict):  # small datasets
            imgs, labels = batch
        else:                            # unlabeled dataset
            imgs = batch["image"]
            labels = batch["label"]
        logits = self.forward(imgs)
        return {
            "logits": logits.detach(), 
            "labels": labels.detach()
        }
    
    def on_train_epoch_end(self):
        if self._step_count > 0:
            mean_loss = self._epoch_loss_sum / self._step_count
            self.train_loss_epoch.append(mean_loss)
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | loss = {mean_loss:.4f}")
    
    def on_validation_epoch_end(self):
        m = self.trainer.callback_metrics.get("ft_val_f1m", None)
        if m is not None:
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | f1_m = {float(m):.4f}")

    def configure_optimizers(self):
        params = []
        if not self.freeze:
            params.append({"params": self.backbone.parameters(), "lr": self.lr_backbone})
        params.append({"params": self.head.parameters(), "lr": self.lr_head})
        
        optim = AdamW(params, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=self.trainer.max_epochs)
        return [optim], [scheduler]
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats
