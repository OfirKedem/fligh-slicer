import pytorch_lightning as pl
import torch

from torch.optim import Adam, SGD


class Regressor(pl.LightningModule):
    def __init__(self, model,
                 loss_fn,
                 lr: float,
                 optim_type: str):
        super().__init__()
        self.model = model
        self.loss = loss_fn

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        optim_type = self.hparams.optim_type
        if optim_type == 'Adam':
            optimizer = Adam(self.parameters(), lr=lr)
        elif optim_type == 'SGD':
            optimizer = SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('Unknown optimizer typr')

        return optimizer
