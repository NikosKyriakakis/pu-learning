import torch
import torch.nn as nn


class PULoss(nn.Module):
    one = torch.tensor(1.)

    def __init__ (
        self, 
        pi_p, 
        gamma=1, 
        beta=0, 
        nn_pu=True, 
        loss_fn=(lambda x: torch.sigmoid(-x))
    ) -> None:

        super().__init__()

        self.pi_p = pi_p
        self.gamma = gamma
        self.beta = beta
        self.loss_fn = loss_fn
        self.nn_pu = nn_pu

    def forward(self, logits, labels):
        # Predictions and labels
        # need to have exactly same dimension 
        assert(logits.shape == labels.shape)

        # Isolate positive & unlabeled samples
        positive = (labels == 1).type(torch.float)
        unlabeled = (labels == -1).type(torch.float)

        # Get total number of positives & unlabeled
        n_positive = torch.max(PULoss.one, torch.sum(positive))
        n_unlabeled = torch.max(PULoss.one, torch.sum(unlabeled))

        # Pass logits through specified loss function
        Rp_plus = self.loss_fn(positive * logits)
        Rp_minus = self.loss_fn(-positive * logits)
        Ru_minus = self.loss_fn(-unlabeled * logits)

        # Calculate positive & negative risk
        positive_risk = self.pi_p * (torch.sum(Rp_plus) / n_positive)
        negative_risk = (torch.sum(Ru_minus) / n_unlabeled) - (self.pi_p * torch.sum(Rp_minus) / n_positive)

        if self.nn_pu and negative_risk < -self.beta:
            loss = -self.gamma * negative_risk
        else:
            loss = positive_risk + negative_risk

        return loss