## Preparing
1. Download pretrain weights and put them in ckpts/

```
# ImageNet-1K pretrained ResNet50, same as torchvision://resnet50
https://cloud.tsinghua.edu.cn/f/3d0cea3f6ac24e019cea/?dl=1
```

2. Create soft link from data/nuscenes to your_nuscenes_path.
The dataset should be organized as follows:

```
TPVFormer/data
    nuscenes                 -    downloaded from www.nuscenes.org
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
    nuscenes_infos_train.pkl
    nuscenes_infos_val.pkl
```

3. Download train/val pickle files and put them in data/
nuscenes_infos_train.pkl
https://cloud.tsinghua.edu.cn/f/ede3023e01874b26bead/?dl=1
nuscenes_infos_val.pkl
https://cloud.tsinghua.edu.cn/f/61d839064a334630ac55/?dl=1
