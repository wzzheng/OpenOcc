
from mmcv.runner import force_fp32, auto_fp16
from mmseg.models import SEGMENTORS

from .base_segmentor import CustomBaseSegmentor

@SEGMENTORS.register_module()
class TPVFormer(CustomBaseSegmentor):

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.fp16_enabled = False

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        B, N, C, H, W = img.shape
        img = img.reshape(B * N, C, H, W)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
        ):
        """Forward training function.
        """
        img_feats = self.extract_img_feat(img=imgs)
        tpv_queries = self.lifter()
        tpv_planes = self.encoder(tpv_queries, img_feats, metas)
        outs = self.head(tpv_planes, points)
        return outs