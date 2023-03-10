
from mmcv.utils import Registry

OPENOCC_LOADER = Registry('openocc_loader')

from .img_pts_loader import ImagePointLoader