import numpy as np
import torch


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
    if dataset_type == 'NuScenes3DOCP':
        results = dict()
        for key, datacontainer in inputs.items():
            results[key] = datacontainer.data[0]
        return results
    else:
        return inputs