import torch
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmseg.models import HEADS

from .base_lifter import BaseLifter


@HEADS.register_module()
class TPVPlainLSSLifter(BaseLifter):

    """ Lift image features to 3D space through naive geometric projection.
    """

    def __init__(self,
                 tpv_h=30,
                 tpv_w=30,
                 tpv_z=30,
                 pc_range=None,
                 **kwargs):
        
        super().__init__()

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.fp16_enabled = False

        self.pc_range = pc_range
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_z = self.pc_range[5] - self.pc_range[2]

        real_voxel_xyz = self.get_reference_points(tpv_h, tpv_w, tpv_z)
        real_voxel_xyz[..., 0] = real_voxel_xyz[..., 0] * self.real_w + pc_range[0]
        real_voxel_xyz[..., 1] = real_voxel_xyz[..., 1] * self.real_h + pc_range[1]
        real_voxel_xyz[..., 2] = real_voxel_xyz[..., 2] * self.real_z + pc_range[2]
        self.register_buffer('real_voxel_xyz', real_voxel_xyz)

    def get_reference_points(self, H, W, Z, device='cpu'):
        xs = torch.linspace(0.5, W - 0.5, W, device=device).view(
            1, -1, 1).expand(H, W, Z) / W
        ys = torch.linspace(0.5, H - 0.5, H, device=device).view(
            -1, 1, 1).expand(H, W, Z) / H
        zs = torch.linspace(0.5, Z - 0.5, Z, device=device).view(
            1, 1, -1).expand(H, W, Z) / Z
        ref_3d = torch.stack((xs, ys, zs), -1) # H, W, Z, 3
        ref_3d = ref_3d.reshape(-1, 3)
        return ref_3d

    @force_fp32(apply_to=('reference_points', 'metas'))
    def point_sampling(self, reference_points, metas):
        lidar2img = []
        for img_meta in metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # hwz, 4

        num_query = reference_points.shape[0]
        B, num_cam = lidar2img.shape[:2]
        reference_points = reference_points.view(
            1, 1, -1, 4, 1).repeat(B, num_cam, 1, 1, 1) # B, N, hwz, 4, 1
        lidar2img = lidar2img.view(
            B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1) # B, N, hwz, 4, 4

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1) # B, N, hwz, 4
        eps = 1e-5

        tpv_mask = (reference_points_cam[..., 2:3] > eps) # B, N, hwz, 1
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], 
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        reference_points_cam[..., 0] /= metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= metas[0]['img_shape'][0][0]

        tpv_mask = (tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        # tpv_mask = torch.nan_to_num(tpv_mask)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            tpv_mask = torch.nan_to_num(tpv_mask)
        else:
            tpv_mask = tpv_mask.new_tensor(
                np.nan_to_num(tpv_mask.cpu().numpy()))

        tpv_mask = tpv_mask.squeeze(-1)

        return reference_points_cam, tpv_mask

    @auto_fp16(apply_to=('ms_img_feats'))
    def forward(self, ms_img_feats, metas):
        # upsample feature maps to the same resolution
        B, N, C, H, W = ms_img_feats[0].shape
        upsampled = [ms_img_feats[0]]
        for feat in ms_img_feats[1:]:
            upsampled.append(
                F.interpolate(
                    feat.flatten(0, 1), (H, W), mode='bilinear', align_corners=True
                ).reshape(B, N, C, H, W))
        
        # concat feature maps along the channel dimension
        concated = torch.cat(upsampled, dim=2) # B, N, 4C, H, W

        # compute voxel-pixel correspondence
        real_voxel_xyz = self.real_voxel_xyz.clone()
        cam_uvs, cam_masks = self.point_sampling(real_voxel_xyz, metas)
        # cam_uvs: B, N, hwz, 2, cam_masks: B, N, hwz
        index_BN = []
        for b in range(B):
            index_N = []
            for n in range(N):
                index_N.append(cam_masks[b, n].nonzero().squeeze(-1))
            index_BN.append(index_N)
        max_len = max([max([len(index) for index in index_N]) for index_N in index_BN])
        cam_uv_tight = torch.ones(
            [B, N, max_len, 2], dtype=cam_uvs.dtype, device=cam_uvs.device) * 2
        for b in range(B):
            for n in range(N):
                index = index_BN[b][n]
                cam_uv_tight[b, n, :len(index), :] = cam_uvs[b, n, index, :]
        
        # grid sample from feature map
        sampled_tight = F.grid_sample(
            concated.reshape(B*N, -1, H, W),
            2 * cam_uv_tight.reshape(B*N, 1, -1, 2) - 1,
            mode='bilinear',
            padding_mode='zeros'
        ).reshape(B, N, -1, max_len) # B, N, 4C, max_len

        sampled = torch.zeros(
            [B, sampled_tight.shape[2], self.tpv_h*self.tpv_w*self.tpv_z], 
            dtype=sampled_tight.dtype, device=sampled_tight.device)
        for b in range(B):
            for n in range(N):
                index = index_BN[b][n]
                sampled[b, :, index] += sampled_tight[b, n, :, :len(index)]
        counts = cam_masks.sum(dim=1, keepdim=True, dtype=sampled.dtype)
        counts = torch.clamp(counts, min=1.0)
        sampled = sampled / counts

        sampled = sampled.reshape(B, sampled.shape[1], self.tpv_h, self.tpv_w, self.tpv_z)
        tpv_hw = sampled.mean(dim=4)
        tpv_zh = sampled.mean(dim=3).permute(0, 1, 3, 2)
        tpv_wz = sampled.mean(dim=2)

        tpv_hw = tpv_hw.permute(0, 2, 3, 1).flatten(1, 2)
        tpv_zh = tpv_zh.permute(0, 2, 3, 1).flatten(1, 2)
        tpv_wz = tpv_wz.permute(0, 2, 3, 1).flatten(1, 2)

        outs = (tpv_hw, tpv_zh, tpv_wz)
        return outs

