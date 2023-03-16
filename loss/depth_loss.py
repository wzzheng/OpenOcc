import torch.nn as nn, torch
import torch.nn.functional as F
from .base_loss import BaseLoss
from . import OPENOCC_LOSS

from torch.cuda.amp.autocast_mode import autocast

@OPENOCC_LOSS.register_module()
class DepthLoss(BaseLoss):

    def __init__(
            self, 
            weight=3.0, 
            downsample_factor=16, 
            dbound=[2.0, 58.0, 0.5], 
            **kwargs):
        super().__init__(weight)
        self.input_keys=['depth_input', 'depth_target']
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.loss_func = self.get_depth_loss

    def get_depth_loss(self, depth_input, depth_target):
        depth_target = self.get_downsampled_gt_depth(depth_target)
        depth_input = depth_input.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_target, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_input[fg_mask],
                depth_target[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

