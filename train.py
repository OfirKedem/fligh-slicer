import comet_ml

import os

import torch.nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from lightning_regressor import Regressor
from models.mlp import MLP
from models.cnn import VGG11
from data.dataset import RoutesDataset


def setup_loader(train_size,
                 val_size,
                 sample_len,
                 batch_size,
                 flatten_channels: bool):
    train_set = RoutesDataset(train_size, sample_len, flatten_channels)
    val_set = RoutesDataset(val_size, sample_len, flatten_channels)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader


def train():
    signal_length = 600
    train_loader, val_loader = setup_loader(train_size=5000,
                                            val_size=100,
                                            sample_len=signal_length,
                                            batch_size=32,
                                            flatten_channels=False)

    # model = MLP(in_size=signal_length * 2)
    model = VGG11()

    lr = 1e-2
    regr = Regressor(model, torch.nn.L1Loss(),
                     lr=lr,
                     optim_type='SGD')

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        save_dir=".",  # Optional
        project_name="flight-slicer",  # Optional
    )

    trainer = pl.Trainer(max_epochs=2,
                         logger=comet_logger,
                         log_every_n_steps=20,
                         )
    trainer.fit(model=regr, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()
