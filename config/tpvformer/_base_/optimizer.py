optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),}
    ),
    weight_decay=0.01
)

ignore_label = 255

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
            ignore=ignore_label)])

grad_max_norm = 35

