from .loader import *
from .modal import *
from .transform import *
from .wrapper import *
from .OCP.builder import build_dataloader_3DOCP

from .utils import custom_collate_fn, convert_inputs

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from mmdet3d.datasets import build_dataset
from copy import deepcopy

def get_dataloader(train_wrapper, val_wrapper, train_loader, val_loader, dist=False):

    train_dataset = OPENOCC_WRAPPER.build(train_wrapper)
    val_dataset = OPENOCC_WRAPPER.build(val_wrapper)

    train_sampler = val_sampler = None
    if dist:
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_dataset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"])
    val_dataset_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"])

    return train_dataset_loader, val_dataset_loader

def get_dataloader_3DOCP(cfg):
    val_dataset = deepcopy(cfg.data.val)
    # in case we use a dataset wrapper
    if 'dataset' in cfg.data.train:
        val_dataset.pipeline = cfg.data.train.dataset.pipeline
    else:
        val_dataset.pipeline = cfg.data.train.pipeline
    # set test_mode=False here in deep copied config
    # which do not affect AP/AR calculation later
    # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
    val_dataset.test_mode = False
    train_dataset, val_dataset = build_dataset(cfg.data.train), build_dataset(val_dataset)
    [train_dataset_loader, val_dataset_loader] = [build_dataloader_3DOCP(
                            train_dataset,
                            cfg.data.samples_per_gpu,
                            cfg.data.workers_per_gpu,
                            # cfg.gpus will be ignored if distributed
                            len(cfg.gpu_ids),
                            dist=cfg.distributed,
                            seed=None,
                            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
                            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
                        ) for bs in [train_dataset, val_dataset]]

    return train_dataset_loader, val_dataset_loader
