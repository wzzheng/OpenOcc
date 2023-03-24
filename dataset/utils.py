import numpy as np
import torch
from mmcv.parallel.data_container import DataContainer


def get_object(item):
    while isinstance(item, (list, tuple)):
        item = item[0]
    return item

def custom_collate_fn(data):
    data_dict = {}

    for item_name, item in data[0].items():

        obj = get_object(item)
        if isinstance(obj, np.ndarray):
            data_dict.update(
                {item_name: torch.from_numpy(np.stack([d[item_name] for d in data]))})
        elif isinstance(item, dict):
            data_dict.update(
                {item_name: [d[item_name] for d in data]})
        elif isinstance(item, torch.Tensor):
            data_dict.update(
                {item_name: torch.stack([d[item_name] for d in data])})
        else:
            raise NotImplementedError
    
    return data_dict

def convert_inputs(inputs, dataset_type):
    if dataset_type == 'NuScenes3DOPC':
        results = dict()
        for key, data in inputs.items():
            if type(data) == DataContainer:
                results[key] = data.data[0]
            else:
                results[key] = data
        return results
    elif dataset_type == 'NuScenes3DOcc':
        results = dict()
        img_aug = dict()
        img_aug_name = ['rots', 'trans', 'intrins', 'post_rots', 'post_trans', 'bda_rot', 'img_shape', 'gt_depths', 'sensor2sensors']
        for key, data in inputs.items():
            if type(data) == DataContainer:
                results[key] = data.data[0]
            elif key == 'img_inputs':
                results[key] = data[0]
                img_shape = data[0].shape
                for i in range(6):
                    img_aug.update({img_aug_name[i]: data[i+1]})
            else:
                results[key] = data
        results['img_metas'][0].update({'img_augmentation': img_aug})
        results['img_metas'][0].update({'img_shape': img_shape})
        return results 
    else:
        return inputs