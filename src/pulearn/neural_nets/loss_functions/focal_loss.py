import torch
import torch.nn.functional as F

from .nnpu_loss import NNPULoss


class FocalNNPULoss(NNPULoss):
    def __init__(
            self,
            prior: float,
            gamma: float,
            beta: float,
            positive_class: int,
            fl_gamma: float = 2.0,
            alpha: float = 0.25
    ) -> None:
        super().__init__(prior=prior, gamma=gamma, beta=beta, positive_class=positive_class)

        self._fl_gamma = fl_gamma
        self._alpha = alpha

    def _surrogate_loss(self, logits: torch.Tensor, labels: torch.Tensor = None):
        p = super()._surrogate_loss(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self._fl_gamma)

        if self._alpha >= 0:
            alpha_t = self._alpha * labels + (1 - self._alpha) * (1 - labels)
            loss = alpha_t * loss

        return loss
