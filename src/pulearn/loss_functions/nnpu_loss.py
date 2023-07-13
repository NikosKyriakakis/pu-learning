import torch
import torch.nn as nn


class NNPULoss(nn.Module):
    def __init__(
            self,
            prior: float,
            gamma: float,
            beta: float,
            positive_class: int
    ) -> None:

        super().__init__()

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.positive_class = positive_class
        self.unlabeled_class = 1 - positive_class

    def _surrogate_loss(self, logits: torch.Tensor, labels: torch.Tensor = None):
        return torch.sigmoid(-logits)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Predictions and labels
        # need to have exactly same dimension
        logits = logits.view(-1)
        assert (logits.shape == labels.shape)

        # Isolate positive & unlabeled samples
        positive = (labels == self.positive_class).float()
        unlabeled = (labels == self.unlabeled_class).float()

        # Get total number of positives & unlabeled
        n_positive = max(1., positive.sum().item())
        n_unlabeled = max(1., unlabeled.sum().item())

        # Pass logits through loss function
        # to get estimates
        y_positive = self._surrogate_loss(logits, labels)
        y_unlabeled = self._surrogate_loss(-logits, labels)

        # Calculate partial risks
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        # Perform gradient ascent to avoid overfitting (nnPU)
        if negative_risk.item() < -self.beta:
            objective = (positive_risk - self.beta + self.gamma * negative_risk).detach() - self.gamma * negative_risk
        # Else perform standard (uPU) learning
        else:
            objective = positive_risk + negative_risk

        return objective
    