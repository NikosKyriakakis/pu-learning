import torch
import torch.nn as nn


class PULoss(nn.Module):
    def __init__ (
        self, 
        prior, 
        gamma=1, 
        beta=0, 
        nn_pu=True, 
        loss_fn=(lambda x: torch.sigmoid(-x))
    ) -> None:

        super().__init__()

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_fn = loss_fn
        self.nn_pu = nn_pu

    def forward(self, logits, labels):

        # Predictions and labels
        # need to have exactly same dimension 
        assert(logits.shape == labels.shape)

        # Isolate positive & unlabeled samples
        positive = (labels == 1)
        unlabeled = (labels == -1)

        # Ensure tensors are of float type
        positive = positive.type(torch.float)
        unlabeled = unlabeled.type(torch.float)

        # Get total number of positives & unlabeled
        min_count = torch.tensor(1.)
        n_positive = torch.max(min_count, torch.sum(positive))
        n_unlabeled = torch.max(min_count, torch.sum(unlabeled))

        # Pass logits through specified loss function
        y_positive = self.loss_fn(logits)
        y_unlabeled = self.loss_fn(-logits)

        # Calculate positive & negative risk
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        if self.nn_pu and negative_risk < -self.beta:
            loss = -self.gamma * negative_risk
        else:
            loss = positive_risk + negative_risk
        
        return loss