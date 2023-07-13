import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from console import error


class NeuralNetEstimator(nn.Module):
    def __init__(
            self,
            pretrained_embeddings: torch.Tensor,
            vocab_len: int,
            embed_dim: int,
            num_classes: int,
            final_activation: nn.Module
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.final_activation = final_activation

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

    def forward(self, x):
        raise NotImplementedError(error("Abstract method called"))


class LstmEstimator(NeuralNetEstimator):
    def __init__(
            self,
            pretrained_embeddings: torch.Tensor,
            vocab_len: int = None,
            embed_dim: int = None,
            num_layers: int = 1,
            hidden_dim: int = 50,
            num_classes: int = 1,
            final_activation: nn.Module = None
    ) -> None:
        super().__init__(pretrained_embeddings, vocab_len, embed_dim, num_classes, final_activation)

        self.lstm = nn.LSTM(
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
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.embedding_layer(x).float()

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self._device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self._device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]

        out = self.linear_in(out)
        out = self.relu(out)
        out = self.linear_out(out)

        if self.final_activation is not None:
            out = self.final_activation(out)

        return out


class CnnEstimator(NeuralNetEstimator):
    def __init__(
            self,
            pretrained_embeddings: torch.Tensor,
            vocab_len: int = None,
            embed_dim: int = None,
            num_filters: int = None,
            filter_sizes: int = None,
            dropout: float = 0.5,
            final_activation = torch.sigmoid,
            num_classes: int = 1,
    ) -> None:
        super().__init__(pretrained_embeddings, vocab_len, embed_dim, num_classes, final_activation)

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        if num_filters is None:
            num_filters = [100, 100, 100]
        self.convolution_modules = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=num_filters[i],
                kernel_size=filter_sizes[i]
            ) for i in range(len(filter_sizes))
        ])

        self.linear_layer = nn.Linear(int(np.sum(num_filters)), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        x = self.embedding_layer(input_ids).float()
        x = x.permute(0, 2, 1)
        x = [F.relu(conv_module(x)) for conv_module in self.convolution_modules]
        x = [F.max_pool1d(x_i, kernel_size=x_i.shape[2]) for x_i in x]
        x = torch.cat([x_i.squeeze(dim=2) for x_i in x], dim=1)
        x = self.linear_layer(self.dropout(x))
        x = self.final_activation(x)

        return x


class MLP5(NeuralNetEstimator):
    def __init__(self,
                 pretrained_embeddings: torch.Tensor,
                 max_document_len,
                 vocab_len: int = None,
                 embed_dim: int = None,
                 num_classes: int = 1,
                 final_activation: nn.Module = None,
                 kernel_size=8
                 ) -> None:
        super().__init__(pretrained_embeddings, vocab_len, embed_dim, num_classes, final_activation)

        input_size = int(self.embed_dim * max_document_len / kernel_size)
        self.input_layer = nn.Linear(input_size, 300)
        self.hidden_layer = nn.Linear(300, 300)
        self.last_layer = nn.Linear(300, self.num_classes)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        x = self.embedding_layer(input_ids).float()
        x = x.view(x.shape[0], -1)
        x = F.avg_pool1d(x, kernel_size=8)
        x = F.softsign(self.input_layer(x))
        x = F.softsign(self.hidden_layer(x))
        x = F.softsign(self.hidden_layer(x))
        x = F.softsign(self.last_layer(x))

        return x


class PUNet(pl.LightningModule):
    def __init__(
            self,
            estimator: NeuralNetEstimator,
            learning_rate: float,
            loss_fn: nn.Module
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.estimator = estimator
        self.loss = loss_fn

    def forward(self, x_in):
        x_out = self.estimator(x_in)
        return x_out

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

        # Uncomment to use with BCE Loss
        logits = logits.squeeze(dim=1)

        loss = self.loss(logits, labels)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sequences, _ = batch
        logits = self(sequences)
        # labels = labels.type(torch.float)
        # preds = torch.sigmoid(logits)
        
        return logits
