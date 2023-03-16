_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/convertion.py'
]
# ================ loss ======================
ignore_label = 0
dbound = [2.0, 58.0, 0.5]

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CELoss',
            weight=1.0,
            ignore_index=ignore_label,),
        dict(
            type='LovaszSoftmaxLoss',
            weight=1.0,
            ignore=ignore_label),
        dict(
            type='DepthLoss',
            weight=3.0,
            downsample_factor=16,
            dbound=dbound),])

# ================== convertion ======================
input_convertion = [
    ['imgs', 'imgs', None, 'cuda'],
    ['points', 'grid_ind_float', 'float', 'cuda'],
    ['point_labels', 'labels', 'long', 'cuda'],
    ['voxel_labels', 'processed_label', 'long', 'cuda'],
    ['depth_target', 'depth_target', None, 'cuda']
]

loss_inputs = dict(
    ce_input='outputs_vox',
    ce_target='voxel_labels',
    lovasz_softmax_input='outputs_pts',
    lovasz_softmax_target='point_labels',
    depth_input='depth_pred',
    depth_target='depth_target'
)

max_num_epochs = 12
load_from = './ckpts/resnet50-0676ba61.pth'

_dim_ = 256
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
        upsample_strides=[0.25, 0.5, 1, 2],
    ),
    lifter=dict(
        type='TPVDepthLSSLifter',
        x_bound=[-51.2, 51.2, 0.512],
        y_bound=[-51.2, 51.2, 0.512],
        z_bound=[-5, 3, 0.5],
        d_bound=dbound,
        final_dim=[480, 800],
        downsample_factor=16,
        output_channels=_dim_,
        depth_net_conf=dict(in_channels=_dim_per_scale_*4, mid_channels=_dim_),
        use_da=False),
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