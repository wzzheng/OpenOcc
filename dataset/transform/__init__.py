
from mmcv.utils import Registry

OPENOCC_TRANSFORM = Registry('openocc_transform')

from .img_transforms import PadMultiViewImage, \
    NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, \
    RandomScaleImageMultiViewImage, \
    DimPermute, \
    GridMask

class TransformCompose(object):

    def __init__(self, cfgs):

        transforms = []
        for cfg in cfgs:
            transforms.append(OPENOCC_TRANSFORM.build(cfg))
        self.transforms = transforms

    def __call__(self, results):

        for t in self.transforms:
            results = t(results)

        return results
    