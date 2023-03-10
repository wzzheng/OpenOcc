from mmseg.models import HEADS
from mmcv.runner import BaseModule


@HEADS.register_module()
class BaseLifter(BaseModule):

    """Base lifter class.
    image backbone -> neck -> lifter -> encoder -> segmentor
    Lift multi-scale image features to 3D representations, e.g. Voxels or TPV or BEV.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(
        self, 
        ms_img_feats, 
        metas=None, 
        **kwargs
    ):
        pass