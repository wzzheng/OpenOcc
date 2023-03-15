import torch.nn as nn
import torch

from .base_lifter import BaseLifter
from mmseg.models import HEADS

@HEADS.register_module()
class TPVQueryLifter(BaseLifter):

    """ Directly initialize tpv features as learnable parameters.
    """

    def __init__(self, tpv_h, tpv_w, tpv_z, embed_dims) -> None:
        super().__init__()
        
        self.tpv_hw = nn.Parameter(torch.randn(tpv_h * tpv_w, embed_dims))
        self.tpv_zh = nn.Parameter(torch.randn(tpv_z * tpv_h, embed_dims))
        self.tpv_wz = nn.Parameter(torch.randn(tpv_w * tpv_z, embed_dims))

    def forward(self,*args, **kwargs):
        return self.tpv_hw, self.tpv_zh, self.tpv_wz