import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DocumentClassifier(pl.LightningModule):
    def __init__(self, pretrained_embedding, embed_dim=300, hidden_size=2, num_layers=1) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.embedding = nn.Embedding.from_pretrained (
            pretrained_embedding
        )

        # Setup LSTM
        self._lstm = nn.LSTM (
            input_size=embed_dim, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True
        ) 
        
        self._fc_1 = nn.Linear(hidden_size, 128) 
        self._fc_2 = nn.Linear(128, 1)

        self._relu = nn.ReLU()

    @property
    def lstm(self):
        return self._lstm

    @property
    def fc_1(self):
        return self._fc_1

    @property
    def relu(self):
        return self._relu

    @property
    def fc_2(self):
        return self._fc_2

    def forward(self, x):
        # Use pretrained embeddings
        x = self.embedding(x).float()
        
        # Calculate initial hidden & internal states
        h_0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size)
        c_0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size)
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h_0, c_0)) 

        # Reshape input
        out = out[:, -1, :]

        # Forward input through 
        # the fully connected layers
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out


    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        # Forward pass
        logits = self(sequences)
        # Loss estimation
        labels = labels.unsqueeze(1).float()
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        labels = labels.unsqueeze(1).float()
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.02)
        return optimizer