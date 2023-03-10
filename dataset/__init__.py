from .loader import *
from .modal import *
from .transform import *
from .wrapper import *

from .utils import custom_collate_fn

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

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
