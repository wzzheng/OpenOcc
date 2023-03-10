import torch.nn as nn
from .base_loss import BaseLoss
from . import OPENOCC_LOSS

from functools import partial
from .utils.lovasz_losses import lovasz_softmax

@OPENOCC_LOSS.register_module()
class LovaszSoftmaxLoss(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight)
        self.input_keys=['lovasz_softmax_input', 'lovasz_softmax_target']
        self.loss_func = partial(lovasz_softmax, **kwargs)
