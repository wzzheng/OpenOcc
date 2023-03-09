# OpenOcc: Easily Extendable 3D Occupancy Prediction Codebase
OpenOcc is an open source 3D occupancy prediction codebase implemented with PyTorch.

# Highlight Features

- **Multiple Benchmarks Support**. 

  We support training and evaluation on different benchmarks including [nuScenes LiDAR Segmentation](https://www.nuscenes.org/lidar-segmentation), [SurroundOcc](https://github.com/weiyithu/SurroundOcc), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), and [3D Occupancy Prediction Challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction). You can even train with sparse lidar supervision and evaluate with dense annotations. :stuck_out_tongue_closed_eyes:

- **Extendable Modular Design.** 

  We design our pipeline to be easily composable and extendable. Feel free to explore other combinations like TPVDepth, VoxelDepth, or TPVFusion with simple modifications. :wink:

# Method

## Dataset 

| Status             | Name | Description                                                  |
| ------------------ | ---- | ------------------------------------------------------------ |
| :white_check_mark: |      | [nuScenes LiDAR Segmentation](https://www.nuscenes.org/lidar-segmentation) |
| :o:                |      | [SurroundOcc](https://github.com/weiyithu/SurroundOcc)       |
| :o:                |      | [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy) |
| :o:                |      | [3D Occupancy Prediction Challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) |

## 2D-3D Lifter

| Status             | Name | 3D Scene Representation | Description                                            |
| ------------------ | ---- | ----------------------- | ------------------------------------------------------ |
| :white_check_mark: |      | TPV                     | Use deformable cross-attention to update TPV queries   |
| :o:                |      | BEV                     | Use deformable cross-attention to update BEV queries   |
| :o:                |      | Voxel                   | Use deformable cross-attention to update Voxel queries |

## Encoder

| Status             | Name | Description                              |
| ------------------ | ---- | ---------------------------------------- |
| :white_check_mark: |      | Use self-attention to aggregate features |
| :o:                |      | Use 2D convolution to aggregate features |
| :o:                |      | Use 3D convolution to aggregate features |

## Loss

| Status             | Name | Description                           |
| ------------------ | ---- | ------------------------------------- |
| :white_check_mark: |      | Cross-entropy loss                    |
| :white_check_mark: |      | [Lovasz-softmax loss](Lovasz-softmax) |
