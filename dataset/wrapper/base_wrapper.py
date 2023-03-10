
from torch.utils import data
from ..loader import OPENOCC_LOADER


class BaseWrapper(data.Dataset):
    """
    Base wrapper for loader to construct a complete dataset class.
    May include transforms applied on images, point cloud or voxels.
    May include voxelization.
    May include dtype convertion.
    """
    
    def __init__(self, loader):
        'Initialization'
        self.loader = OPENOCC_LOADER.build(loader)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        data = self.loader[index]

        return data
