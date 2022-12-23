import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from console import *

from pulearn\
    .neural_nets\
    .loss_functions\
    .nnpu_loss import *


class NeuralNetEstimator(nn.Module):
    def __init__(self, pretrained_embeddings, vocab_len, embed_dim, num_classes=1) -> None:
        super().__init__()

        self.num_classes = num_classes

        if pretrained_embeddings is None \
            and type(vocab_len) == int \
            and type(embed_dim) == int:
            self.embedding_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)
            self.embed_dim = embed_dim
            self.vocab_len = vocab_len
        else:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings)
            self.vocab_len = pretrained_embeddings.shape[0]
            self.embed_dim = pretrained_embeddings.shape[1]

    def forward(self, X):
        raise NotImplementedError(error("Abstract method called"))


class LSTM_Estimator(NeuralNetEstimator):
    def __init__(
        self, 
        pretrained_embeddings, 
        vocab_len=None, 
        embed_dim=None,
        num_layers=1,
        hidden_dim=50,
        num_classes=1
    ) -> None:

        super().__init__(pretrained_embeddings, vocab_len, embed_dim, num_classes)
        
        self.lstm = nn.LSTM (
            input_size=self.embed_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True
        ) 

        self.num_layers = num_layers
        self.hidden_size = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, 128) 
        self.linear_out = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.embedding_layer(X).float()
        
        # Calculate initial hidden & internal states
        h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        # Propagate input through LSTM
        out, _ = self.lstm(X, (h_0, c_0)) 

        # Reshape input
        out = out[:, -1, :]

        # Forward input through 
        # the fully connected layers
        out = self.linear_in(out)
        out = self.relu(out)
        out = self.linear_out(out)

        return out


class CNN_Estimator(NeuralNetEstimator):
    def __init__(
        self,  
        pretrained_embeddings, 
        vocab_len=None,
        embed_dim=None,
        num_filters=[100, 100, 100], 
        filter_sizes=[3, 4, 5],
        dropout=0.5,
        apply_softmax=False,
        num_classes=1,
    ) -> None:

        super().__init__(pretrained_embeddings, vocab_len, embed_dim, num_classes)

        self.convolution_modules = nn.ModuleList ([
            nn.Conv1d(
                in_channels=self.embed_dim, 
                out_channels=num_filters[i], 
                kernel_size=filter_sizes[i]
            ) for i in range(len(filter_sizes))
        ])

        self.linear_layer = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self._apply_softmax = apply_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids):
        X = self.embedding_layer(input_ids).float()
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