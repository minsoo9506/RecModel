import argparse

import torch
from torchmetrics import Accuracy
from torchmetrics.regression import MeanSquaredError

from .lit_base import LitBase

OPTIMIZER = "Adam"
LOSS = "MSELoss"
LR = 0.001


class LitTwoTower(LitBase):
    def __init__(self, model, args: argparse.Namespace) -> None:
        super().__init__(model, args)

        if self.args.get("loss", LOSS) == "MSELoss":
            self.train_metric = MeanSquaredError()
            self.valid_metric = MeanSquaredError()
            self.test_metric = MeanSquaredError()
        if self.args.get("loss", LOSS) == "BCELoss":
            self.train_metric = Accuracy("binary")
            self.valid_metric = Accuracy("binary")
            self.test_metric = Accuracy("binary")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        y, preds, loss = self._run_on_batch(batch)
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
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        y, preds, loss = self._run_on_batch(batch)
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
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        y, preds, loss = self._run_on_batch(batch)
        self.test_metric(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/metric", self.valid_metric, on_step=False, on_epoch=True)
