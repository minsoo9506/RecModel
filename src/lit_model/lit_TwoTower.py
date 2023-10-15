import argparse
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torchmetrics.regression import MeanSquaredError

OPTIMIZER = "Adam"
LOSS = "MSELoss"
LR = 0.001


class LitTwoTower(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace) -> None:
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn, loss)()

        if loss == "MSELoss":
            self.train_metric = MeanSquaredError()
            self.valid_metric = MeanSquaredError()
            self.test_metric = MeanSquaredError()
        if loss == "BCELoss":
            self.train_metric = Accuracy("binary")
            self.valid_metric = Accuracy("binary")
            self.test_metric = Accuracy("binary")

    def forward(
        self,
        x: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "monitor": "validation/loss",
        }

    def _run_on_batch_two_tower(
        self,
        batch: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, candidate, y = batch
        preds = self((query, candidate))
        loss = self.loss_fn(preds, y)
        return y, preds, loss

    def training_step(
        self,
        batch: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        y, preds, loss = self._run_on_batch_two_tower(batch)
        self.train_metric(preds, y)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "train/metric",
            self.train_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss}

    def validation_step(
        self,
        batch: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> None:
        y, preds, loss = self._run_on_batch_two_tower(batch)
        self.valid_metric(preds, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "validation/metric",
            self.valid_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(
        self,
        batch: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> None:
        y, preds, loss = self._run_on_batch_two_tower(batch)
        self.test_metric(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/metric", self.valid_metric, on_step=False, on_epoch=True)
