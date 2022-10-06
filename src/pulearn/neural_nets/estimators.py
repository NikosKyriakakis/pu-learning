import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pulearn.neural_nets.nnpu_loss import PULoss


class CNNEstimator(nn.Module):
    def __init__(
        self, 
        num_classes, 
        pretrained_embedding, 
        num_filters=[100, 100, 100], 
        filter_sizes=[3, 4, 5],
        dropout=0.5
    ) -> None:

        super().__init__()

        self.embeeding = nn.Embedding.from_pretrained (
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

    def forward(self, input_ids):
        X = self.embeeding(input_ids).float()
        X = X.permute(0, 2, 1)
        X = [F.relu(conv_module(X)) for conv_module in self.convolution_modules]
        X = [F.max_pool1d(x_i, kernel_size=x_i.shape[2]) for x_i in X]
        X = torch.cat([x_i.squeeze(dim=2) for x_i in X], dim=1)
        X = self.linear_layer(self.dropout(X))

        return X


class PUNet(pl.LightningModule):
    def __init__(
        self,
        estimator,
        learning_rate,
        pi_p=0.5, 
        gamma=1, 
        beta=0, 
        nn_pu=True, 
        loss_fn=(lambda x: torch.sigmoid(-x))
    ) -> None:

        super().__init__()
    
        self.learning_rate = learning_rate
        self.estimator = estimator

        self.pi_p = pi_p
        self.gamma = gamma
        self.beta = beta
        self.loss_fn = loss_fn
        self.nn_pu = nn_pu

    def forward(self, X_in):
        X_out = self.estimator(X_in)
        return X_out

    def run_step(self, batch, stage):
        sequences, labels = batch
        logits = self(sequences)
        labels = labels.type(torch.float)

        loss_fct = PULoss(pi_p=self.pi_p, gamma=self.gamma, beta=self.beta, nn_pu=self.nn_pu, loss_fn=self.loss_fn)
        loss = loss_fct(logits.view(-1), labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
