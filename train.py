import comet_ml

import os

import torch.nn
import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_regressor import Regressor
from models.model_registry import models
from data.dataset import RoutesDataset


def setup_loader(train_size,
                 val_size,
                 sample_len,
                 batch_size,
                 flatten_channels: bool,
                 num_workers: int = 2,
                 **_kwargs):
    train_set = RoutesDataset(train_size, sample_len, flatten_channels)
    val_set = RoutesDataset(val_size, sample_len, flatten_channels)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


def train(cfg: dict):
    pl.seed_everything(cfg['seed'])

    train_loader, val_loader = setup_loader(**cfg['data'],
                                            flatten_channels=True)

    arch = cfg['arch']
    model = models[arch['type']](**arch['kwargs'])

    regr = Regressor(model,
                     torch.nn.L1Loss(),
                     **cfg['train'])

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        save_dir=".",
        project_name="flight-slicer",
        experiment_name=cfg['experiment_name']
    )

    callbacks = [ModelCheckpoint(monitor='val/loss')]

    trainer = pl.Trainer(max_epochs=2,
                         logger=comet_logger,
                         log_every_n_steps=20,
                         gpus=1 if torch.cuda.is_available() else 0,
                         callbacks=callbacks
                         )
    trainer.fit(model=regr, train_dataloaders=train_loader, val_dataloaders=val_loader)

    comet_logger.experiment.end()


if __name__ == "__main__":
    cfg_path = 'train_config.yaml'
    config = yaml.safe_load(open(cfg_path, 'r'))
    train(config)
