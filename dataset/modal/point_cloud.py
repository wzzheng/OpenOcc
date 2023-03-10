import numpy as np

from .base_modal import BaseModal
from .utils import nb_process_label

class PointCloud(BaseModal):

    def __init__(self, grid_size, 
                 fill_label=0, 
                 max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5]) -> None:

        super().__init__()
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.max_bound = np.asarray(max_volume_space)
        self.min_bound = np.asarray(min_volume_space)

    def to_voxel(self, points, labels):
        # get grid index
        crop_range = self.max_bound - self.min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        intervals = crop_range / cur_grid_size

        if (intervals == 0).any(): 
            print("Zero interval!")
        grid_ind_float = (np.clip(
            points, self.min_bound, self.max_bound - 1e-3) - self.min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        return grid_ind_float, None, processed_label
