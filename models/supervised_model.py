import torch
from torch import Tensor

from torch.nn import CrossEntropyLoss, Linear
from torch.optim import AdamW
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC


class Supervised(LightningModule):
    def __init__(
            self, 
            backbone,
            h_params,
            cls_weights,
            freeze=False
    ):
        super().__init__()
        n_classes = len(cls_weights)
        self.lr: float = h_params["lr"]
        
        self.backbone = backbone
        feat_dim = self.backbone.feat_dim
        self.head = Linear(feat_dim, n_classes)
        
        self.loss_function = CrossEntropyLoss(weight=Tensor(cls_weights))
        self.train_losses = []
        self.train_loss_epoch = []
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

        # Global accuracy over all samples, standard overall top-1 accuracy
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        # f1 per class, then average
        self.f1_multi = MulticlassF1Score(num_classes=n_classes, average="macro")
        self.auroc = MulticlassAUROC(num_classes=n_classes, average="macro")

        self.freeze_encoder = freeze
        if self.freeze_encoder:
            self.backbone.freeze()
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch).flatten(start_dim=1)
        output = self.head(feats)
        return output
    
    def _calculate_loss(self, batch, mode: str) -> torch.Tensor:
        imgs, labels = batch
        outputs = self.forward(imgs)
        
        # calculate loss and metrics
        loss = self.loss_function(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        f1_score = self.f1_multi(outputs, labels)
        auroc = self.auroc(torch.softmax(outputs, dim=1), labels)
        
        # log loss and metrics
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{mode}_f1m", f1_score, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{mode}_auroc", auroc, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch):
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
    
    def validation_step(self, batch, _=0):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, _=0):
        return self._calculate_loss(batch, mode="test")

    def predict_step(self, batch, _=0, __=0):
        x, y = batch
        logits = self(x)
        return {"logits": logits, "labels": y}

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.lr)
        return optim
    
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(batch)
        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)
        return feats
