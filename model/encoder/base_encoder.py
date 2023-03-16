from mmseg.models import HEADS
from mmcv.runner import BaseModule


@HEADS.register_module()
class BaseEncoder(BaseModule):
    """Further encode 3D representations.
    image backbone -> neck -> lifter -> encoder -> segmentor
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(
        self, 
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        pass