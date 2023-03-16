import numpy as np

from .base_modal import BaseModal
from .utils import nb_process_label

class PointCloud(BaseModal):

    def __init__(
        self, 
        to_voxel_args=dict(
            grid_size=[200, 200, 16], 
            fill_label=0, 
            max_volume_space=[51.2, 51.2, 3], 
            min_volume_space=[-51.2, -51.2, -5]),
        to_depthmap_args=dict(
            downsample=16,)
    ) -> None:

        super().__init__()
        self.to_voxel_args = dict(
            grid_size = np.asarray(to_voxel_args['grid_size']), 
            fill_label = to_voxel_args['fill_label'], 
            max_bound = np.asarray(to_voxel_args['max_volume_space']), 
            min_bound = np.asarray(to_voxel_args['min_volume_space']))
        
        self.to_depthmap_args = dict(
            downsample = to_depthmap_args['downsample'])

    def to_voxel(self, points, labels):
        # get grid index
        max_bound = self.to_voxel_args['max_bound']
        min_bound = self.to_voxel_args['min_bound']
        grid_size = self.to_voxel_args['grid_size']
        fill_label = self.to_voxel_args['fill_label']
        crop_range = max_bound - min_bound
        intervals = crop_range / grid_size

        if (intervals == 0).any(): 
            print("Zero interval!")
        grid_ind_float = (np.clip(
            points, min_bound, max_bound - 1e-3) - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int)

        # process labels
        processed_label = np.ones(grid_size, dtype=np.uint8) * fill_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        return grid_ind_float, None, processed_label
    
    def to_depth_map(self, points, lidar2img, img_shape, ida_args=None):
        """
        points: N, 3
        lidar2img: C, 4, 4
        """
        num_points = points.shape[0]
        num_cams = lidar2img.shape[0]

        lidar2img = np.reshape(lidar2img, (num_cams, 1, 4, 4))
        points = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=1)
        points = np.reshape(points, (1, num_points, 4, 1))
        points = np.squeeze(lidar2img @ points, axis=-1)[..., :3] # C, N, 3

        points[..., :2] = points[..., :2] / points[..., 2:3]

        mask = np.ones(points.shape[:2], dtype=bool) # C, N
        mask = np.logical_and(mask, points[..., 2] > 0)
        mask = np.logical_and(mask, points[..., 0] > 1)
        mask = np.logical_and(mask, points[..., 0] < img_shape[1] - 1)
        mask = np.logical_and(mask, points[..., 1] > 1)
        mask = np.logical_and(mask, points[..., 1] < img_shape[0] - 1)

        depth_maps = []
        for cam in range(num_cams):
            cam_depth = points[cam][mask[cam]] # n, 3
            # cam_depth[:, :2] = cam_depth[:, :2] * resize
            # cam_depth[:, 0] -= crop[0]
            # cam_depth[:, 1] -= crop[1]
            # if flip:
                # cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

            # cam_depth[:, 0] -= W / 2.0
            # cam_depth[:, 1] -= H / 2.0

            # h = rotate / 180 * np.pi
            # rot_matrix = [
                # [np.cos(h), np.sin(h)],
                # [-np.sin(h), np.cos(h)],
            # ]
            # cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

            # cam_depth[:, 0] += W / 2.0
            # cam_depth[:, 1] += H / 2.0

            depth_coords = cam_depth[:, :2].astype(np.int16)

            depth_map = np.zeros(img_shape)
            valid_mask = ((depth_coords[:, 1] < img_shape[0])
                        & (depth_coords[:, 0] < img_shape[1])
                        & (depth_coords[:, 1] >= 0)
                        & (depth_coords[:, 0] >= 0))
            depth_map[depth_coords[valid_mask, 1],
                    depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

            depth_maps.append(depth_map)
        
        return np.stack(depth_maps) # C, H, W

