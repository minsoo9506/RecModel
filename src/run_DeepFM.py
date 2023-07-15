import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataload.load_data import DeepFMDataset
from lit_model.DeepFM_lit_model import DeepFMLitModel
from model.DeepFM import DeepFM

N_SAMPLES = 2000


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="DeepFM")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=16,
        help="embedding dimensions (default: 16)",
    )
    parser.add_argument(
        "--mlp_dims",
        default=[16, 16],
        help="mlp hidden layers' dimensions (default: [16, 16])",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 3)"
    )
    parser.add_argument("--cuda", type=int, default=0, help="0 for cpu -1 for all gpu")
    config = parser.parse_args()
    if config.cuda == 0 or torch.cuda.is_available() is False:
        config.cuda = 0

    return config


def main(config):
    # data
    data_path = "/home/minsoo/Workspace/RecSys-study/data/kmrd/kmr_dataset/datafile/kmrd-small/rates.csv"  # kmrd-small data
    data = pd.read_csv(data_path, 0.2)

    data = data[:N_SAMPLES]

    train_ratio = 0.8
    DeepFM_dataset = DeepFMDataset(data)
    train_dataset, valid_dataset = random_split(
        DeepFM_dataset, [train_ratio, 1.0 - train_ratio]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    # model
    DeepFM_model = DeepFM(
        field_dims=DeepFM_dataset.field_dims,
        embed_dim=config.embed_dim,
        mlp_dims=config.mlp_dims,
        dropout=0.2,
    )

    DeepFM_lit_model = DeepFMLitModel(DeepFM_model, config)

    # trainer
    # logger = pl.loggers.WandbLogger()
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="validation/loss", mode="min", patience=20
    )
    trainer = pl.Trainer(
        # logger=logger,
        log_every_n_steps=10,  # set the logging frequency
        gpus=config.cuda,  # use all GPUs
        max_epochs=config.epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        callbacks=[early_stopping_callback],
    )

    # fit the model
    trainer.fit(DeepFM_lit_model, train_loader, valid_loader)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
