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
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.530, 116.280, 123.675], 
            std=[1.0, 1.0, 1.0], 
            to_rgb=False),
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
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.530, 116.280, 123.675], 
            std=[1.0, 1.0, 1.0], 
            to_rgb=False),
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

max_epochs = 12
load_from = './ckpts/resnet50-0676ba61.pth'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 128
num_heads = 8
_pos_dim_ = [48, 48, 32]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6
scale_rate = 0.5

tpv_h_ = 200
tpv_w_ = 200
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
nbr_class = 18

# ============= MODEL ===============

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_,
            ),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)

tpv_encoder_layers = [
    self_cross_layer, 
    self_cross_layer,
    self_cross_layer,   
]

model = dict(
    type='TPVFormer',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    lifter=dict(
        type='TPVQueryLifter',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        embed_dims=_dim_,),
    encoder=dict(
        type='TPVFormerEncoder',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        embed_dims=_dim_,
        num_cams=_num_cams_,
        num_feature_levels=_num_levels_,
        pc_range=point_cloud_range,
        positional_encoding=dict(
            type='TPVPositionalEncoding',
            num_feats=_pos_dim_,
            h=tpv_h_,
            w=tpv_w_,
            z=tpv_z_),
        num_points_in_pillar=num_points_in_pillar,
        num_points_in_pillar_cross_view=[16, 16, 16],
        num_layers=len(tpv_encoder_layers),
        transformerlayers=tpv_encoder_layers),
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