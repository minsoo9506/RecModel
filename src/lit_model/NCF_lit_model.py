import argparse
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch

OPTIMIZER = "Adam"
LOSS = "MSELoss"
LR = 0.001


class NCFLitModel(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace = None):
        """_summary_

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

    def _run_on_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        return loss

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
        loss = self._run_on_batch(batch)

        self.log("train/loss", loss)
        # BCELoss 사용시 아래처럼 ACC 볼 수 있게
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

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
        loss = self._run_on_batch(batch)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)

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
        loss = self._run_on_batch(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
