# Modified from BEVDepth
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import build_conv_layer
from mmseg.models.backbones.resnet import BasicBlock
from torch import nn, torch
from torch.cuda.amp.autocast_mode import autocast

from .base_lifter import BaseLifter
from mmseg.models import HEADS

try:
    from .ops.voxel_pooling_inference import voxel_pooling_inference
    from .ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

__all__ = ['TPVDepthLSSLifter']


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        
        meta_channels = 16

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(meta_channels)
        self.depth_mlp = Mlp(meta_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(meta_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrinsic'][:, :, :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[1]
        # ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['cam2lidar'][:, :, :3, :]
        # bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        # 4).repeat(1, 1, num_cams, 1, 1)
        
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, ..., 0, 0],
                        intrins[:, ..., 1, 1],
                        intrins[:, ..., 0, 2],
                        intrins[:, ..., 1, 2],
                        # ida[:, 0:1, ..., 0, 0],
                        # ida[:, 0:1, ..., 0, 1],
                        # ida[:, 0:1, ..., 0, 3],
                        # ida[:, 0:1, ..., 1, 0],
                        # ida[:, 0:1, ..., 1, 1],
                        # ida[:, 0:1, ..., 1, 3],
                        # bda[:, 0:1, ..., 0, 0],
                        # bda[:, 0:1, ..., 0, 1],
                        # bda[:, 0:1, ..., 1, 0],
                        # bda[:, 0:1, ..., 1, 1],
                        # bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


@HEADS.register_module()
class TPVDepthLSSLifter(BaseLifter):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 depth_net_conf,
                 use_da=False):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super().__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = self._configure_depth_aggregation_net()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        print(points.shape)
        if ida_mat is not None:
            ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = ida_mat.inverse().matmul(points)
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_feats,
                              mats_dict,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_feats (Tensor): Image feats.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                # sensor2sensor_mats(Tensor): Transformation matrix
                #     from key frame camera to sweep frame camera with
                #     shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: TPV (BEV) feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_feats.shape
        source_features = sweep_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        depth = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(
            mats_dict['cam2lidar'],
            mats_dict['intrinsic'],
            # mats_dict['img2lidar'],
            mats_dict.get('ida_mats', None),
            mats_dict.get('bda_mat', None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        if self.training or self.use_da:
            img_feat_with_depth = depth.unsqueeze(
                1) * depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
            
            voxel_num_cuda = self.voxel_num.cuda()
            tpv_hw = voxel_pooling_train(
                geom_xyz,
                img_feat_with_depth.contiguous(),
                voxel_num_cuda)
            tpv_zh = voxel_pooling_train(
                geom_xyz[..., [1, 2, 0]],
                img_feat_with_depth.contiguous(),
                voxel_num_cuda[[1, 2, 0]])
            tpv_wz = voxel_pooling_train(
                geom_xyz[..., [2, 0, 1]],
                img_feat_with_depth.contiguous(),
                voxel_num_cuda[[2, 0, 1]])
            
        else:
            voxel_num_cuda = self.voxel_num.cuda()
            tpv_hw = voxel_pooling_inference(
                geom_xyz, 
                depth, 
                depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].contiguous(),
                voxel_num_cuda)
            tpv_zh = voxel_pooling_inference(
                geom_xyz[..., [1, 2, 0]], 
                depth, 
                depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].contiguous(),
                voxel_num_cuda[[1, 2, 0]])
            tpv_wz = voxel_pooling_inference(
                geom_xyz[..., [2, 0, 1]], 
                depth, 
                depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].contiguous(),
                voxel_num_cuda[[2, 0, 1]])
        tpv = (tpv_hw, tpv_zh, tpv_wz)

        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return tpv, depth_feature[:, :self.depth_channels].softmax(dim=1)
        return tpv
    
    def rearrange_metas(self, metas, dtype, device):
        cam2lidars = []
        intrinsics = []
        for meta in metas:
            cam2lidars.append(meta['cam2lidar'])
            intrinsics.append(meta['intrinsic'])
        
        cam2lidars = torch.from_numpy(np.stack(cam2lidars)).to(device=device, dtype=dtype)
        intrinsics = torch.from_numpy(np.stack(intrinsics)).to(device=device, dtype=dtype)

        return {'cam2lidar': cam2lidars, 'intrinsic': intrinsics}

    def forward(self,
                ms_img_feats,
                metas,
                timestamps=None,
                is_return_depth=True,
                **kwargs):
        """Forward function.

        Args:
            ms_img_feats(List[Tensor]): Image feats with shape of (B,
                num_cameras, C, H, W).
            metas(List[dict]):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_cameras, 4, 4).
                # sensor2sensor_mats(Tensor): Transformation matrix
                #     from key frame camera to sweep frame camera with
                #     shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            # timestamps(Tensor): Timestamp for all images with the shape of(B,
            #     num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        metas = self.rearrange_metas(metas, ms_img_feats[0].dtype, ms_img_feats[0].device)
        num_sweeps = 1
        # num_sweeps = ms_img_feats[0].shape[1]
        # batch_size, num_cams, num_channels, img_height, \
        #     img_width = ms_img_feats[0].shape

        key_frame_res = self._forward_single_sweep(
            0,
            ms_img_feats[0].unsqueeze(1),
            metas,
            is_return_depth=is_return_depth)
        assert num_sweeps == 1
        if num_sweeps == 1:
            if is_return_depth:
                outs = {'representation': key_frame_res[0], 'depth_pred': key_frame_res[1]}
            else:
                outs = {'representation': key_frame_res[0]}
            return outs

        # key_frame_feature = key_frame_res[
        #     0] if is_return_depth else key_frame_res

        # ret_feature_list = [key_frame_feature]
        # for sweep_index in range(1, num_sweeps):
        #     with torch.no_grad():
        #         feature_map = self._forward_single_sweep(
        #             sweep_index,
        #             ms_img_feats[0][:, sweep_index:sweep_index + 1, ...],
        #             metas[sweep_index],
        #             is_return_depth=False)
        #         ret_feature_list.append(feature_map)

        # if is_return_depth:
        #     return torch.cat(ret_feature_list, 1), key_frame_res[1]
        # else:
        #     return torch.cat(ret_feature_list, 1)
