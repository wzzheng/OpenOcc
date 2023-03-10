import torch.nn as nn
from .base_loss import BaseLoss
from . import OPENOCC_LOSS

@OPENOCC_LOSS.register_module()
class CELoss(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight)
        
        self.input_keys = ['ce_input', 'ce_target']
        self.loss_func = nn.CrossEntropyLoss(**kwargs)
