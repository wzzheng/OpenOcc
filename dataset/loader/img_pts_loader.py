import os, numpy as np, yaml
from mmcv.image.io import imread
from nuscenes import NuScenes

from .base_loader import BaseLoader
from . import OPENOCC_LOADER

@OPENOCC_LOADER.register_module()
class ImagePointLoader(BaseLoader):
    def __init__(
        self, 
        data_path, 
        pkl_path='./data/nuscenes_infos_train.pkl', 
        label_mapping="./config/label_mapping/nuscenes.yaml", 
        nusc=None,
        version=None,
        return_img=True,
        return_pts=True,
    ):
        super().__init__(data_path, pkl_path)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.nuScenes_label_name = self.get_nuScenes_label_name(nuscenesyaml)
        if nusc is None:
            nusc = NuScenes(version=version, dataroot=data_path)
        self.nusc = nusc
        self.return_img = return_img
        self.return_pts = return_pts

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        metas = self.get_data_info(info)
        imgs = points = points_label = None
        
        if self.return_img:
            imgs = [] # read 6 cams
            for filename in metas['img_filename']:
                imgs.append(imread(filename, 'unchanged').astype(np.float32))

        if self.return_pts:
            lidar_sd_token = self.nusc.get('sample', metas['sample_token'])['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(
                self.data_path, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
            points_label = points_label.astype(np.uint8)
            
            lidar_path = metas['pts_filename']
            points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
            points = points[:, :3]

        data_tuple = (imgs, metas, points, points_label)
        return data_tuple
    
    def get_data_info(self, info):
        input_dict = dict(
            sample_token=info['token'],
            pts_filename=info['lidar_path'],)

        image_paths = []
        lidar2img_rts = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

        input_dict.update(dict(
            img_filename=image_paths,
            lidar2img=lidar2img_rts,))
        return input_dict

    def get_nuScenes_label_name(self, nuScenesyaml):
        nuScenes_label_name = dict()
        for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
            val_ = nuScenesyaml['learning_map'][i]
            nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]
        return nuScenes_label_name
