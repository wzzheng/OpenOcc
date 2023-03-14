
from mmcv.runner import force_fp32, auto_fp16
from mmseg.models import SEGMENTORS

from .base_segmentor import CustomBaseSegmentor

@SEGMENTORS.register_module()
class TPVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.fp16_enabled = False

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
        ):
        """Forward training function.
        """
        img_feats = self.extract_img_feat(img=imgs)
        tpv = self.lifter(img_feats, metas)
        tpv = self.encoder(tpv, img_feats, metas)
        outs = self.head(tpv, points)
        return outs