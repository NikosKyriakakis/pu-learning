import torch
import torch.nn.functional as F

from .nnpu_loss import NNPULoss


def ramp_loss(x: torch.Tensor):
    return torch.max(torch.tensor(0.), torch.min(torch.tensor(1.), (torch.tensor(1.) - x) / torch.tensor(2)))

class RampLossNNPU(NNPULoss):
    def __init__(
            self,
            prior: float,
            gamma: float,
            beta: float,
            positive_class: int
    ) -> None:
        super().__init__(prior=prior, gamma=gamma, beta=beta, positive_class=positive_class)

    def _surrogate_loss(self, logits: torch.Tensor, labels: torch.Tensor = None):
        loss = ramp_loss(logits)
        return loss
