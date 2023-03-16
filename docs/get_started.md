## Getting Started

### Training

1. Train TPVFormer for lidar segmentation task.
```
bash launcher.sh config/tpvformer/tpvformer_lidarseg_dim128_r50_800.py out/tpvformer_lidarseg_dim128_r50_800
```

2. Train TPVConv with PlainLSSLifter for lidar segmentation task.
```
bash launcher.sh config/tpvconv/tpvconv_lidarseg_dim384_r50_800_layer10.py out/tpvconv_lidarseg_dim384_r50_800_layer10
```

3. Train TPVConv with DepthLSSLifter for lidar segmentation task.
```
bash launcher.sh config/tpvconv/tpvconv_lidarseg_dim384_r50_800_layer10_depthlss.py out/tpvconv_lidarseg_dim384_r50_800_layer10_depthlss
```
