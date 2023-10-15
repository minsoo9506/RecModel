import argparse

import torch
from torchmetrics import Accuracy

from .lit_base import LitBase

OPTIMIZER = "Adam"
LOSS = "MSELoss"
LR = 0.001


class LitNeuMF(LitBase):
<<<<<<< HEAD
    def __init__(self, model, args: argparse.Namespace = None):
=======
    def __init__(self, model, args: argparse.Namespace):
>>>>>>> develop
        super().__init__(model, args)

        self.train_acc = Accuracy("binary")
        self.valid_acc = Accuracy("binary")
        self.test_acc = Accuracy("binary")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        y, preds, loss = self._run_on_batch(batch)
        self.train_acc(preds, y)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return {"loss": loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        y, preds, loss = self._run_on_batch(batch)
        self.valid_acc(preds, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "validation/acc",
            self.valid_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        y, preds, loss = self._run_on_batch(batch)
        self.test_acc(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.valid_acc, on_step=False, on_epoch=True)
