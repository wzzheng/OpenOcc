
from mmcv.utils import Registry

OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .ce_loss import CELoss
from .lovasz_loss import LovaszSoftmaxLoss