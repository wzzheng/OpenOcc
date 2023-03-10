data_path = 'data/nuscenes'
label_mapping = "config/label_mapping/nuscenes.yaml"
version = 'v1.0-trainval'

unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
metric_ignore_label = 0

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
            type='PadMultiViewImage',
            size_divisor=32),
        dict(
            type='DimPermute',
            permute_order=[2, 0, 1],)],
    pointcloudModal = pointcloudModal)

train_loader = dict(
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

val_loader = dict(
    batch_size=1,
    num_workers=1,
)