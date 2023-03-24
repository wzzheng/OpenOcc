
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

    @auto_fp16(apply_to=('imgs', 'points'))
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
        ):
        """Forward training function.
        """
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }
        outs = self.extract_img_feat(**results)
        results.update(outs)
        outs = self.lifter(**results)
        results.update(outs)
        outs = self.encoder(**results)
        results.update(outs)
        outs = self.head(**results)
        results.update(outs)
        return results