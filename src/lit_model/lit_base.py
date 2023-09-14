import argparse
from typing import Any

import pytorch_lightning as pl
import torch

# from torchmetrics import Accuracy

OPTIMIZER = "Adam"
LOSS = "BCELoss"
LR = 0.001


class LitBase(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn, loss)()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        # if self.one_cycle_max_lr is None:
        #     return optimizer
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     max_lr=self.one_cycle_max_lr,
        #     total_steps=self.one_cycle_total_steps,
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "validation/loss",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.model(x)
        return preds

    def _run_on_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return y, preds, loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        y, preds, loss = self._run_on_batch(batch)
        # self.train_acc(preds, y)

        self.log("train/loss", loss)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        y, preds, loss = self._run_on_batch(batch)
        # self.valid_acc(preds, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        # self.log("validation/acc", self.valid_acc, on_step=False, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        y, preds, loss = self._run_on_batch(batch)
        # self.test_acc(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", self.valid_acc, on_step=False, on_epoch=True)
