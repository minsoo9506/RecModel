import argparse
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

OPTIMIZER = "Adam"
LOSS = "BCELoss"
LR = 0.001


class DeepFMLitModel(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace = None):
        """DeepFM Lit Model (BCELoss)

        Parameters
        ----------
        model : _type_
            NCF model
        args : argparse.Namespace, optional
            _description_, by default None
        """
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn, loss)()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    # @staticmethod
    # def add_to_argparse(parser):
    #     parser.add_argument(
    #         "--optimizer",
    #         type=str,
    #         default=OPTIMIZER,
    #         help="optimizer class from torch.optim",
    #     )
    #     parser.add_argument("--lr", type=float, default=LR)
    #     parser.add_argument("--one_cycle_max_lr", type=float, default=None)
    #     parser.add_argument(
    #         "--loss",
    #         type=str,
    #         default=LOSS,
    #         help="loss function from torch.nn.functional",
    #     )
    #     return parser

    def configure_optimizers(self) -> Dict[str, Any]:
        """set optimizer

        Returns
        -------
        Dict[str, Any]
            opimizer, validation loss
        """
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
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward propagation

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            x input, y label

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            y label, y pred, loss
        """
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        return y, preds, loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """training step

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            batch data
        batch_idx : int
            batch index

        Returns
        -------
        Dict[str, torch.Tensor]
            {"loss": loss}
        """
        y, preds, loss = self._run_on_batch(batch)
        self.train_acc(preds, y)

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """validation step

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            batch data
        batch_idx : int
            batch index

        Returns
        -------
        Dict[str, torch.Tensor]
            {"loss": loss}
        """
        y, preds, loss = self._run_on_batch(batch)
        self.valid_acc(preds, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/acc", self.valid_acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """test step

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            batch data
        batch_idx : int
            batch index
        """
        y, preds, loss = self._run_on_batch(batch)
        self.test_acc(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.valid_acc, on_step=False, on_epoch=True)
