import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")

from nuscenes import NuScenesExplorer

class CustomNuScenes:
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-trainval',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1,
                 lidarseg_path: str = '/data/sets/nuscenes',
                 map_path: str = '/data/sets/nuscenes'
        ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)
        if osp.exists(lidarseg_path):
            self.lidarseg_table_path = osp.join(lidarseg_path, version)
            self.category = self.__load_table_lidarseg__('category')
            has_lidarseg_task = True
        else:
            self.category = self.__load_table__('category')
            has_lidarseg_task = False

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        # self.category = self.__load_table__('category')

        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        lidar_tasks = [t for t in ['panoptic'] if osp.exists(osp.join(self.table_root, t + '.json'))]
        if has_lidarseg_task:
            lidar_tasks += ['lidarseg']
        if len(lidar_tasks) > 0:
            self.lidarseg_idx2name_mapping = dict()
            self.lidarseg_name2idx_mapping = dict()
            self.load_lidarseg_cat_name_mapping()
        for i, lidar_task in enumerate(lidar_tasks):
            if self.verbose:
                print(f'Loading nuScenes-{lidar_task}...')
            if lidar_task == 'lidarseg':
                self.lidarseg = self.__load_table_lidarseg__(lidar_task)
                tmp_data_root = lidarseg_path
            else:
                self.panoptic = self.__load_table__(lidar_task)
                tmp_data_root = self.dataroot

            # setattr(self, lidar_task, self.__load_table__(lidar_task))
            label_files = os.listdir(os.path.join(tmp_data_root, lidar_task, self.version))
            num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
            num_lidarseg_recs = len(getattr(self, lidar_task))
            assert num_lidarseg_recs == num_label_files, \
                f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
            self.table_names.append(lidar_task)
            # Sort the colormap to ensure that it is ordered according to the indices in self.category.
            self.colormap = dict({c['name']: self.colormap[c['name']]
                                  for c in sorted(self.category, key=lambda k: k['index'])})

        # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        if osp.exists(osp.join(self.table_root, 'image_annotations.json')):
            self.image_annotations = self.__load_table__('image_annotations')

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record['mask'] = MapMask(osp.join(map_path, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)
    
    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table
    
    def __load_table_lidarseg__(self, table_name):
        with open(osp.join(self.lidarseg_table_path, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def load_lidarseg_cat_name_mapping(self):
        """ Create mapping from class index to class name, and vice versa, for easy lookup later on """
        for lidarseg_category in self.category:
            # Check that the category records contain both the keys 'name' and 'index'.
            assert 'index' in lidarseg_category.keys(), \
                'Please use the category.json that comes with nuScenes-lidarseg, and not the old category.json.'

            self.lidarseg_idx2name_mapping[lidarseg_category['index']] = lidarseg_category['name']
            self.lidarseg_name2idx_mapping[lidarseg_category['name']] = lidarseg_category['index']

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """
        This function queries all records for a certain field value, and returns the tokens for the matching records.
        Warning: this runs in linear time.
        :param table_name: Table name.
        :param field: Field name. See README.md for details.
        :param query: Query to match against. Needs to type match the content of the query field.
        :return: List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return osp.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [self.get('sample_annotation', token) for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def get_sample_lidarseg_stats(self,
                                  sample_token: str,
                                  sort_by: str = 'count',
                                  lidarseg_preds_bin_path: str = None,
                                  gt_from: str = 'lidarseg') -> None:
        """
        Print the number of points for each class in the lidar pointcloud of a sample. Classes with have no
        points in the pointcloud will not be printed.
        :param sample_token: Sample token.
        :param sort_by: One of three options: count / name / index. If 'count`, the stats will be printed in
                        ascending order of frequency; if `name`, the stats will be printed alphabetically
                        according to class name; if `index`, the stats will be printed in ascending order of
                        class index.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param gt_from: 'lidarseg' or 'panoptic', ground truth source of point semantic labels.
        """
        assert gt_from in ['lidarseg', 'panoptic'], f'gt_from can only be lidarseg or panoptic, get {gt_from}'
        assert hasattr(self, gt_from), f'Error: You have no {gt_from} data; unable to get ' \
                                       'statistics for segmentation of the point cloud.'
        assert sort_by in ['count', 'name', 'index'], 'Error: sort_by can only be one of the following: ' \
                                                      'count / name / index.'
        semantic_table = getattr(self, gt_from)
        sample_rec = self.get('sample', sample_token)
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_record = self.get('sample_data', ref_sd_token)

        # Ensure that lidar pointcloud is from a keyframe.
        assert ref_sd_record['is_key_frame'], 'Error: Only pointclouds which are keyframes have ' \
                                              'lidar segmentation labels. Rendering aborted.'

        if lidarseg_preds_bin_path:
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} ' \
                '(lidar sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, ref_sd_token)

            header = '===== Statistics for ' + sample_token + ' (predictions) ====='
        else:
            assert len(semantic_table) > 0, 'Error: There are no ground truth labels found for nuScenes-{} for {}.'\
                                            'Are you loading the test set? \nIf you want to see the sample statistics'\
                                            ' for your predictions, pass a path to the appropriate .bin/npz file using'\
                                            ' the lidarseg_preds_bin_path argument.'.format(gt_from, self.version)
            lidar_sd_token = self.get('sample', sample_token)['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(self.dataroot,
                                                    self.get(gt_from, lidar_sd_token)['filename'])

            header = '===== Statistics for ' + sample_token + ' ====='
        print(header)

        points_label = load_bin_file(lidarseg_labels_filename, type=gt_from)
        if gt_from == 'panoptic':
            points_label = panoptic_to_lidarseg(points_label)
        lidarseg_counts = get_stats(points_label, len(self.lidarseg_idx2name_mapping))

        lidarseg_counts_dict = dict()
        for i in range(len(lidarseg_counts)):
            lidarseg_counts_dict[self.lidarseg_idx2name_mapping[i]] = lidarseg_counts[i]

        if sort_by == 'count':
            out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1])
        elif sort_by == 'name':
            out = sorted(lidarseg_counts_dict.items())
        else:
            out = lidarseg_counts_dict.items()

        for class_name, count in out:
            if count > 0:
                idx = self.lidarseg_name2idx_mapping[class_name]
                print('{:3}  {:40} n={:12,}'.format(idx, class_name, count))

        print('=' * len(header))

    def list_categories(self) -> None:
        self.explorer.list_categories()

    def list_lidarseg_categories(self, sort_by: str = 'count', gt_from: str = 'lidarseg') -> None:
        self.explorer.list_lidarseg_categories(sort_by=sort_by, gt_from=gt_from)

    def list_panoptic_instances(self, sort_by: str = 'count', get_hist: bool = False) -> None:
        self.explorer.list_panoptic_instances(sort_by=sort_by, get_hist=get_hist)

    def list_attributes(self) -> None:
        self.explorer.list_attributes()

    def list_scenes(self) -> None:
        self.explorer.list_scenes()

    def list_sample(self, sample_token: str) -> None:
        self.explorer.list_sample(sample_token)

    def render_pointcloud_in_image(self, sample_token: str, dot_size: int = 5, pointsensor_channel: str = 'LIDAR_TOP',
                                   camera_channel: str = 'CAM_FRONT', out_path: str = None,
                                   render_intensity: bool = False,
                                   show_lidarseg: bool = False,
                                   filter_lidarseg_labels: List = None,
                                   show_lidarseg_legend: bool = False,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_panoptic: bool = False) -> None:
        self.explorer.render_pointcloud_in_image(sample_token, dot_size, pointsensor_channel=pointsensor_channel,
                                                 camera_channel=camera_channel, out_path=out_path,
                                                 render_intensity=render_intensity,
                                                 show_lidarseg=show_lidarseg,
                                                 filter_lidarseg_labels=filter_lidarseg_labels,
                                                 show_lidarseg_legend=show_lidarseg_legend,
                                                 verbose=verbose,
                                                 lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                 show_panoptic=show_panoptic)

    def render_sample(self, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      show_lidarseg: bool = False,
                      filter_lidarseg_labels: List = None,
                      lidarseg_preds_bin_path: str = None,
                      verbose: bool = True,
                      show_panoptic: bool = False) -> None:
        self.explorer.render_sample(sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path,
                                    show_lidarseg=show_lidarseg, filter_lidarseg_labels=filter_lidarseg_labels,
                                    lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=verbose,
                                    show_panoptic=show_panoptic)

    def render_sample_data(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 40, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None, verbose: bool = True,
                           show_panoptic: bool = False) -> None:
        self.explorer.render_sample_data(sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         underlay_map=underlay_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         show_lidarseg=show_lidarseg,
                                         show_lidarseg_legend=show_lidarseg_legend,
                                         filter_lidarseg_labels=filter_lidarseg_labels,
                                         lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                         verbose=verbose,
                                         show_panoptic=show_panoptic)

    def render_annotation(self, sample_annotation_token: str, margin: float = 10, view: np.ndarray = np.eye(4),
                          box_vis_level: BoxVisibility = BoxVisibility.ANY, out_path: str = None,
                          extra_info: bool = False) -> None:
        self.explorer.render_annotation(sample_annotation_token, margin, view, box_vis_level, out_path, extra_info)

    def render_instance(self, instance_token: str, margin: float = 10, view: np.ndarray = np.eye(4),
                        box_vis_level: BoxVisibility = BoxVisibility.ANY, out_path: str = None,
                        extra_info: bool = False) -> None:
        self.explorer.render_instance(instance_token, margin, view, box_vis_level, out_path, extra_info)

    def render_scene(self, scene_token: str, freq: float = 10, imsize: Tuple[float, float] = (640, 360),
                     out_path: str = None) -> None:
        self.explorer.render_scene(scene_token, freq, imsize, out_path)

    def render_scene_channel(self, scene_token: str, channel: str = 'CAM_FRONT', freq: float = 10,
                             imsize: Tuple[float, float] = (640, 360), out_path: str = None) -> None:
        self.explorer.render_scene_channel(scene_token, channel=channel, freq=freq, imsize=imsize, out_path=out_path)

    def render_egoposes_on_map(self, log_location: str, scene_tokens: List = None, out_path: str = None) -> None:
        self.explorer.render_egoposes_on_map(log_location, scene_tokens, out_path=out_path)

    def render_scene_channel_lidarseg(self, scene_token: str,
                                      channel: str,
                                      out_folder: str = None,
                                      filter_lidarseg_labels: Iterable[int] = None,
                                      with_anns: bool = False,
                                      render_mode: str = None,
                                      verbose: bool = True,
                                      imsize: Tuple[int, int] = (640, 360),
                                      freq: float = 2,
                                      dpi: int = 150,
                                      lidarseg_preds_folder: str = None,
                                      show_panoptic: bool = False) -> None:
        self.explorer.render_scene_channel_lidarseg(scene_token,
                                                    channel,
                                                    out_folder=out_folder,
                                                    filter_lidarseg_labels=filter_lidarseg_labels,
                                                    with_anns=with_anns,
                                                    render_mode=render_mode,
                                                    verbose=verbose,
                                                    imsize=imsize,
                                                    freq=freq,
                                                    dpi=dpi,
                                                    lidarseg_preds_folder=lidarseg_preds_folder,
                                                    show_panoptic=show_panoptic)

    def render_scene_lidarseg(self, scene_token: str,
                              out_path: str = None,
                              filter_lidarseg_labels: Iterable[int] = None,
                              with_anns: bool = False,
                              imsize: Tuple[int, int] = (640, 360),
                              freq: float = 2,
                              verbose: bool = True,
                              dpi: int = 200,
                              lidarseg_preds_folder: str = None,
                              show_panoptic: bool = False) -> None:
        self.explorer.render_scene_lidarseg(scene_token,
                                            out_path=out_path,
                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                            with_anns=with_anns,
                                            imsize=imsize,
                                            freq=freq,
                                            verbose=verbose,
                                            dpi=dpi,
                                            lidarseg_preds_folder=lidarseg_preds_folder,
                                            show_panoptic=show_panoptic)