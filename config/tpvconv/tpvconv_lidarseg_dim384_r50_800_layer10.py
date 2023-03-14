_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/convertion.py'
]

data_path = 'data/nuscenes'
label_mapping = "config/label_mapping/nuscenes.yaml"
version = 'v1.0-trainval'

pointcloudModal = dict(
        grid_size=[200, 200, 16],
        fill_label=0, 
        max_volume_space=[51.2, 51.2, 3], 
        min_volume_space=[-51.2, -51.2, -5])

train_wrapper = dict(
    type='ImagePointWrapper',
    loader = dict(
        type='ImagePointLoader',
        data_path=data_path,
        pkl_path='data/nuscenes_infos_train.pkl', 
        label_mapping=label_mapping, 
        nusc=None,
        version=version,
        return_img=True,
        return_pts=True,),
    img_transforms = [
        dict(
            type='PhotoMetricDistortionMultiViewImage'),
        # dict(
        #     type='NormalizeMultiviewImage',
        #     mean=[103.530, 116.280, 123.675], 
        #     std=[1.0, 1.0, 1.0], 
        #     to_rgb=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375], 
            to_rgb=True),
        dict(
            type='RandomScaleImageMultiViewImage',
            scales=[0.5]),
        dict(
            type='PadMultiViewImage',
            size_divisor=32),
        dict(
            type='DimPermute',
            permute_order=[2, 0, 1],),
        dict(
            type='GridMask',
            use_h=True, 
            use_w=True, 
            rotate=1, 
            offset=False, 
            ratio=0.5, 
            mode=1, 
            prob=0.7)],
    pointcloudModal = pointcloudModal)

val_wrapper = dict(
    type='ImagePointWrapper',
    loader = dict(
        type='ImagePointLoader',
        data_path=data_path,
        pkl_path='data/nuscenes_infos_val.pkl', 
        label_mapping=label_mapping, 
        nusc=None,
        version=version,
        return_img=True,
        return_pts=True,),
    img_transforms = [
        # dict(
        #     type='NormalizeMultiviewImage',
        #     mean=[103.530, 116.280, 123.675], 
        #     std=[1.0, 1.0, 1.0], 
        #     to_rgb=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375], 
            to_rgb=True),
        dict(
            type='RandomScaleImageMultiViewImage',
            scales=[0.5]),
        dict(
            type='PadMultiViewImage',
            size_divisor=32),
        dict(
            type='DimPermute',
            permute_order=[2, 0, 1],)],
    pointcloudModal = pointcloudModal)

max_num_epochs = 12
load_from = './ckpts/resnet50-0676ba61.pth'

_dim_ = 384
_dim_per_scale_ = 96
tpv_h_ = 200
tpv_w_ = 200
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
nbr_class = 17
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

syncBN = True

# ============= MODEL ===============
model = dict(
    type='TPVSegmentor',
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(1,2,3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_eval=False,
        style='pytorch',),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=[_dim_per_scale_] * 4,
        upsample_strides=[0.5, 1, 2, 4],
        # type='FPN',
        # in_channels=[512, 1024, 2048],
        # out_channels=_dim_,
        # start_level=0,
        # add_extra_convs='on_output',
        # num_outs=4,
        # relu_before_extra_convs=True
    ),
    lifter=dict(
        type='TPVPlainLSSLifter',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,),
    encoder=dict(
        type='TPVConvEncoder',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        feature=_dim_,
        bottleneck_layer='dense',
        norm_layer='naive',
        feature_downsample=4,
        expansion=4,
        dilations=[1, 2, 3, 1]
        ),
    head=dict(
        type='TPVHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z),)