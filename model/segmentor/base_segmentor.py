from mmcv.runner import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmseg.models import SEGMENTORS, builder
from mmdet3d.models import build_neck

@SEGMENTORS.register_module()
class CustomBaseSegmentor(BaseModule):

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if img_backbone is not None:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            try:
                self.img_neck = builder.build_neck(img_neck)
            except:
                self.img_neck = build_neck(img_neck)
        if lifter is not None:
            self.lifter = builder.build_head(lifter)
        if encoder is not None:
            self.encoder = builder.build_head(encoder)
        if head is not None:
            self.head = builder.build_head(head)

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        B, N, C, H, W = img.size()
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

    def forward(
        self,
        imgs,
        metas,
        **kwargs
    ):
        pass