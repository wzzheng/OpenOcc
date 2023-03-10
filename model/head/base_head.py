from mmcv.runner import BaseModule
from mmseg.models import HEADS

@HEADS.register_module()
class BaseSegmentor(BaseModule):
    """Segmentation heads.
    image backbone -> neck -> lifter -> encoder -> segmentor
    Predicts semantic labels for voxels (and points for lidar segmentation).
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(
        self, 
        representation,
        points=None,
        **kwargs
    ):
        pass