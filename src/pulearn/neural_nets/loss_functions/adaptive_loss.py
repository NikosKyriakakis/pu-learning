from robust_loss_pytorch import AdaptiveLossFunction

import torch


class AdaptiveLossFunctionMod(AdaptiveLossFunction):
    """ 
        - This is an extension of the adaptive robust loss 
        pytorch port found here: https://github.com/jonbarron/robust_loss_pytorch

         - Here we define a forward method and play around with the dimensions of the predictions/targets
        to make the base class compatible with libraries such as skorch, etc ...
    """
    def __init__(self, num_dims, float_dtype, device, alpha_lo=0.001, alpha_hi=1.999, alpha_init=None, scale_lo=0.00001, scale_init=1):
        super().__init__(num_dims, float_dtype, device, alpha_lo, alpha_hi, alpha_init, scale_lo, scale_init)

    def forward(self, predictions, targets):
        X = torch.cat((predictions, targets.unsqueeze(dim=1)), dim=1)
        X = torch.mean(self.lossfun(X))

        return X