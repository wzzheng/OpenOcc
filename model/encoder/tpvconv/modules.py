import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptedConv(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1, bias=True
    ):
        super().__init__()
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(padding, int):
            padding = [padding] * 3
        if isinstance(dilation, int):
            dilation = [dilation] * 3
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.use_bias = bias

        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, groups=groups)

        k = math.sqrt(groups / in_channels / kernel_size[0] / kernel_size[1] / kernel_size[2])
        weight = torch.rand([out_channels, int(in_channels/groups), *kernel_size])
        weight = weight * 2 * k - k
        if self.use_bias:
            bias = torch.rand(out_channels)
            bias = bias * 2 * k - k
            self.bias = nn.Parameter(bias)
        self.weight = nn.Parameter(weight)

    # def forward(self, tpv_hw, tpv_zh, tpv_wz):
    def forward(self, tpv):
        tpv_hw, tpv_zh, tpv_wz = tpv

        weight = self.weight # out, in/groups, h, w, z
        bias = self.bias if self.use_bias else None # out

        hw_weight = weight.sum(dim=4) # out, in/groups, h, w
        zh_weight = weight.sum(dim=3).permute(0, 1, 3, 2) # out, in/groups, z, h
        wz_weight = weight.sum(dim=2) # out, in/groups, w, z

        tpv_hw = F.conv2d(
            tpv_hw, hw_weight, bias, 
            self.stride[:2], self.padding[:2], self.dilation[:2], self.groups)
        tpv_zh = F.conv2d(
            tpv_zh, zh_weight, bias, 
            self.stride[-1::-2], self.padding[-1::-2], self.dilation[-1::-2], self.groups)
        tpv_wz = F.conv2d(
            tpv_wz, wz_weight, bias, 
            self.stride[1:], self.padding[1:], self.dilation[1:], self.groups)
        
        return tpv_hw, tpv_zh, tpv_wz


class AdaptedBN3d(nn.Module):

    def __init__(
        self,
        num_features,
        momentum=0.1,
        eps=1e-5
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones([1, num_features, 1, 1])
        )
        self.beta = nn.Parameter(
            torch.zeros([1, num_features, 1, 1])
        )

        mean_hw = torch.zeros([1, num_features, 1, 1])
        mean_zh = torch.zeros([1, num_features, 1, 1])
        mean_wz = torch.zeros([1, num_features, 1, 1])
        std = torch.ones([1, num_features, 1, 1])
        self.register_buffer('mean_hw', mean_hw)
        self.register_buffer('mean_zh', mean_zh)
        self.register_buffer('mean_wz', mean_wz)
        self.register_buffer('std', std)
    
    # def forward(self, tpv_hw, tpv_zh, tpv_wz):
    def forward(self, tpv):
        tpv_hw, tpv_zh, tpv_wz = tpv

        if self.training:
            H, W = tpv_hw.shape[-2:]
            Z = tpv_zh.shape[-2]
            # tpv_hw: B, C, H, W  tpv_zh: B, C, Z, H  tpv_wz: B, C, W, Z
            mean_hw = tpv_hw.mean(dim=[0, 2, 3], keepdim=True) # 1, C, 1, 1
            mean_zh = tpv_zh.mean(dim=[0, 2, 3], keepdim=True)
            mean_wz = tpv_wz.mean(dim=[0, 2, 3], keepdim=True)

            # cal_std
            cent_hw = tpv_hw - mean_hw
            cent_zh = tpv_zh - mean_zh
            cent_wz = tpv_wz - mean_wz
            # hw
            cent_zh_hw = cent_zh.mean(dim=2).unsqueeze(-1).expand(-1, -1, -1, W)
            cent_wz_hw = cent_wz.mean(dim=3).unsqueeze(2).expand(-1, -1, H, -1)
            std_hw = (cent_hw * (cent_hw + cent_zh_hw + cent_wz_hw)).mean(
                dim=[0, 2, 3], keepdim=True)
            # zh
            cent_hw_zh = cent_hw.mean(dim=3).unsqueeze(2).expand(-1, -1, Z, -1)
            cent_wz_zh = cent_wz.mean(dim=2).unsqueeze(-1).expand(-1, -1, -1, H)
            std_zh = (cent_zh * (cent_hw_zh + cent_zh + cent_wz_zh)).mean(
                dim=[0, 2, 3], keepdim=True)
            # wz
            cent_hw_wz = cent_hw.mean(dim=2).unsqueeze(-1).expand(-1, -1, -1, Z)
            cent_zh_wz = cent_zh.mean(dim=3).unsqueeze(2).expand(-1, -1, W, -1)
            std_wz = (cent_wz * (cent_hw_wz + cent_zh_wz + cent_wz)).mean(
                dim=[0, 2, 3], keepdim=True)
            
            std = std_hw + std_zh + std_wz # 1, C, 1, 1
 
            # update features
            multiplier = self.gamma / torch.sqrt(std + self.eps)

            # update running mean and vars
            self.mean_hw.data = (1 - self.momentum) * self.mean_hw.data + \
                self.momentum * mean_hw.detach().data
            self.mean_zh.data = (1 - self.momentum) * self.mean_zh.data + \
                self.momentum * mean_zh.detach().data
            self.mean_wz.data = (1 - self.momentum) * self.mean_wz.data + \
                self.momentum * mean_wz.detach().data
            self.std.data = (1 - self.momentum) * self.std.data + \
                self.momentum * std.detach().data
        else:
            cent_hw = tpv_hw - self.mean_hw
            cent_zh = tpv_zh - self.mean_zh
            cent_wz = tpv_wz - self.mean_wz
            multiplier = self.gamma / torch.sqrt(self.std + self.eps)
        
        tpv_hw = cent_hw * multiplier + self.beta
        tpv_zh = cent_zh * multiplier + self.beta
        tpv_wz = cent_wz * multiplier + self.beta
        
        return tpv_hw, tpv_zh, tpv_wz


class NaiveBN3d(nn.Module):
    
    def __init__(
        self,
        num_features,
        momentum=0.1,
        eps=1e-5
    ):
        super().__init__()
        self.bn_hw = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn_zh = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn_wz = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, tpv):
        tpv_hw, tpv_zh, tpv_wz = tpv
        tpv_hw = self.bn_hw(tpv_hw)
        tpv_zh = self.bn_zh(tpv_zh)
        tpv_wz = self.bn_wz(tpv_wz)
        
        return tpv_hw, tpv_zh, tpv_wz


class AdaptedReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    # def forward(self, tpv_hw, tpv_zh, tpv_wz):
    def forward(self, tpv):
        tpv_hw, tpv_zh, tpv_wz = tpv

        tpv_hw = self.relu(tpv_hw)
        tpv_zh = self.relu(tpv_zh)
        tpv_wz = self.relu(tpv_wz)
        return tpv_hw, tpv_zh, tpv_wz