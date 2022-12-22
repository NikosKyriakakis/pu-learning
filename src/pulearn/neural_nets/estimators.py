import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pulearn\
    .neural_nets\
    .loss_functions\
    .nnpu_loss import *


class MLP5(nn.Module):
    def __init__(self, pretrained_embedding, max_len, kernel_size=8) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained (
            pretrained_embedding
        )

        embed_size = pretrained_embedding.shape[1]
        input_size = int((embed_size * max_len) / kernel_size)
        self.input_layer = nn.Linear(input_size, 300)
        self.hidden_layer = nn.Linear(300, 300)
        self.last_layer = nn.Linear(300, 2)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids):
        X = self.embedding(input_ids).float()
        X = X.view(X.shape[0], -1)
        X = F.avg_pool1d(X, kernel_size=8) 
        X = F.softsign(self.input_layer(X))
        X = self.dropout(X)
        X = self.softmax(self.last_layer(X))
        
        return X


class CNNEstimator(nn.Module):
    def __init__(
        self, 
        num_classes, 
        pretrained_embedding, 
        num_filters=[100, 100, 100], 
        filter_sizes=[3, 4, 5],
        dropout=0.5,
        apply_softmax=False
    ) -> None:

        super().__init__()

        self.embedding = nn.Embedding.from_pretrained (
            pretrained_embedding
        )

        dim = pretrained_embedding.shape[1]

        self.convolution_modules = nn.ModuleList ([
            nn.Conv1d(
                in_channels=dim, 
                out_channels=num_filters[i], 
                kernel_size=filter_sizes[i]
            ) for i in range(len(filter_sizes))
        ])

        self.linear_layer = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self._apply_softmax = apply_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids):
        X = self.embedding(input_ids).float()
        X = X.permute(0, 2, 1)
        X = [F.relu(conv_module(X)) for conv_module in self.convolution_modules]
        X = [F.max_pool1d(x_i, kernel_size=x_i.shape[2]) for x_i in X]
        X = torch.cat([x_i.squeeze(dim=2) for x_i in X], dim=1)
        X = self.linear_layer(self.dropout(X))

        if self._apply_softmax:
            X = self.softmax(X)
     
        return X


class PUNet(pl.LightningModule):
    def __init__ (
        self, 
        estimator, 
        learning_rate
    ) -> None:

        super().__init__()

        self.learning_rate = learning_rate
        self.estimator = estimator

    def forward(self, X_in):
        X_out = self.estimator(X_in)
        return X_out

    def run_step(self, batch, stage):
        raise NotImplementedError()
        
    def training_step(self, batch, batch_idx):
        return self.run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        return optimizer

    def run_step(self, batch, stage):
        sequences, labels = batch
        logits = self(sequences)
        labels = labels.type(torch.float)

        loss = self.loss(logits.view(-1), labels)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss


class NNPUNet(PUNet):
    def __init__(
        self,
        estimator,
        learning_rate,
        positive_class=1,
        prior=0.5, 
        gamma=1, 
        beta=0,
        loss_fn=(lambda x: torch.sigmoid(-x))
    ) -> None:

        super().__init__(estimator, learning_rate)

        self.loss = NNPULoss (
            prior=prior, 
            gamma=gamma, 
            beta=beta, 
            loss_fn=loss_fn, 
            positive_class=positive_class
        )


class AAPUNet(PUNet):
    def __init__(
        self,
        estimator,
        learning_rate,
        positive_class=1,
        prior=0.5, 
        gamma=1, 
        beta=0,
        loss_fn=(lambda x: logloss(-x))
    ) -> None:

        super().__init__(estimator, learning_rate)

        self.loss = NNPULoss (
            prior=prior, 
            gamma=gamma, 
            beta=beta, 
            loss_fn=loss_fn, 
            positive_class=positive_class
        )