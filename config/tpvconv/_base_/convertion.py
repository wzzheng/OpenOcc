
input_convertion = [
    ['imgs', 'imgs', None, 'cuda'],
    ['points', 'grid_ind_float', 'float', 'cuda'],
    ['point_labels', 'labels', None, 'cuda'],
    ['voxel_labels', 'processed_label', 'long', 'cuda'],
]

model_inputs = dict(
    imgs='imgs',
    metas='metas',
    points='points'
)

loss_inputs = dict(
    ce_input='outputs_vox',
    ce_target='voxel_labels',
    lovasz_softmax_input='outputs_pts',
    lovasz_softmax_target='point_labels'
)