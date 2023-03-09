# OpenOcc: Easily Extendable 3D Occupancy Prediction Pipeline
Open Source 3D Occupancy Prediction Library.

## Highlight Features

- **Multiple Benchmarks Support**. 

  We support training and evaluation on different benchmarks including [nuScenes LiDAR Segmentation](https://www.nuscenes.org/lidar-segmentation), [SurroundOcc](https://github.com/weiyithu/SurroundOcc), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), and [3D Occupancy Prediction Challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction). You can even train with sparse lidar supervision and evaluate with dense annotations. :stuck_out_tongue_closed_eyes:

- **Extendable Modular Design.** 

  We design our pipeline to be easily composable and extendable. Feel free to explore other combinations like TPVDepth, VoxelDepth, or TPVFusion with simple modifications of configs. :wink:

<div align="center">
  <b>Main Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Dataset</b>
      </td>
      <td>
        <b>2D-3D Lifter</b>
      </td>
      <td>
        <b>Encoder</b>
      </td>
      <td>
        <b>Head</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="https://www.nuscenes.org/lidar-segmentation">nuScenes LiDAR Segmentation</a></li>
            <li><a href="https://github.com/weiyithu/SurroundOcc"><i>SurroundOcc</i></a></li>
            <li><a href="https://github.com/JeffWang987/OpenOccupancy"><i>OpenOccupancy</i></a></li>
            <li><a href="https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction"><i>3D Occupancy Prediction Challenge</i></a></li>
      </ul>
      </td>
      <td>
      </ul>
          <li><b>BEV</b></li>
        <ul>
        <ul>
          <li><a href="https://arxiv.org/abs/2203.17270"><i>Attention</i></a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Voxel</b></li>
        <ul>
        <ul>
          <li><a href="https://github.com/weiyithu/SurroundOcc"><i>Attention</i></a></li>
        </ul>
        </ul>
      </ul>
        <li><b>TPV</b></li>
      <ul>
        <ul>
          <li><a href="https://wzzheng.net/TPVFormer/">Attention</a></li>
        </ul>
        </ul>
      </ul>
      </td>
      <td>
        <ul>
          <li>Attention</li>
          <li><i>2D Convolution</i></li>
          <li><i>3D Convolution</i></li>
        </ul>
      </td>
      <td>
         <ul>
          <li>FC</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>Cross-entropy</li>
            <li><a href="https://arxiv.org/abs/1705.08790">Lovasz-softmax</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>
