import torch.nn as nn
from mmseg.models import HEADS
from .modules import AdaptedConv, AdaptedBN3d, AdaptedReLU, NaiveBN3d
from ..base_encoder import BaseEncoder

class TPVBottleneck3D_sparse(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        bn_momentum=0.0003,
    ):
        super().__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = AdaptedConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = AdaptedConv(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = AdaptedConv(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = AdaptedConv(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = AdaptedConv(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = AdaptedReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        # out3 = out3 + out2
        out3 = [o3 + o2 for o3, o2 in zip(out3, out2)]
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        # out4 = out4 + out2 + out3
        out4 = [o4 + o2 + o3 for o4, o2, o3 in zip(out4, out2, out3)]

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = out5 + residual
        out = [o5 + r for o5, r in zip(out5, residual)]
        out_relu = self.relu(out)

        return out_relu


class TPVBottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        bn_momentum=0.0003,
    ):
        super().__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = AdaptedConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = AdaptedConv(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = AdaptedConv(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = AdaptedReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out5 = self.bn5(self.conv5(out2_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = out5 + residual
        out = [o5 + r for o5, r in zip(out5, residual)]
        out_relu = self.relu(out)

        return out_relu


class TPVProcess(nn.Module):
    def __init__(
        self, feature, bottleneck_layer, norm_layer, bn_momentum, dilations=[1, 2, 3], 
        feature_downsample=4, expansion=4):
        super().__init__()
        self.main = nn.Sequential(
            *[
                bottleneck_layer(
                    feature,
                    feature // feature_downsample,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                    expansion=expansion
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)

class TPVChannelAdaptor(nn.Module):
    def __init__(self, feature, bottleneck_layer, norm_layer, bn_momentum, feature_downsample=4, expansion=4):
        super().__init__()
        self.main = bottleneck_layer(
            feature,
            feature // feature_downsample,
            bn_momentum=bn_momentum,
            norm_layer=norm_layer,
            expansion=expansion,
            downsample=nn.Sequential(
                AdaptedConv(
                    feature,
                    int(feature * expansion / feature_downsample),
                    kernel_size=1,
                    stride=1,
                    bias=False,),
                norm_layer(int(feature * expansion / feature_downsample), momentum=bn_momentum),
            ))

    def forward(self, x):
        return self.main(x)


@HEADS.register_module()
class TPVConvEncoder(BaseEncoder):
    def __init__(
        self,
        tpv_h, tpv_w, tpv_z,
        feature,
        bottleneck_layer='sparse',
        norm_layer='adapted',
        bn_momentum=0.1,
        feature_downsample=4,
        expansion=2,
        dilations=[1, 2, 3, 1],
        **kwargs
    ):
        super().__init__()
        
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z

        self.feature = feature
        if norm_layer == 'adapted':
            norm_layer = AdaptedBN3d
        elif norm_layer == 'naive':
            norm_layer = NaiveBN3d
        else:
            raise NotImplementedError
        
        if bottleneck_layer == 'sparse':
            bottleneck_layer = TPVBottleneck3D_sparse
        elif bottleneck_layer == 'dense':
            bottleneck_layer = TPVBottleneck3D
        else:
            raise NotImplementedError

        self.process_l1 = TPVProcess(
            feature, bottleneck_layer, norm_layer, bn_momentum, dilations, feature_downsample, feature_downsample)
        self.channel_adaptor1 = TPVChannelAdaptor(
            feature, bottleneck_layer, norm_layer, bn_momentum, feature_downsample, expansion)
        feature_1 = int(feature / feature_downsample * expansion)

        self.process_l2 = TPVProcess(
            feature_1, bottleneck_layer, norm_layer, bn_momentum, dilations, feature_downsample, feature_downsample)
        self.channel_adaptor2 = TPVChannelAdaptor(
            feature_1, bottleneck_layer, norm_layer, bn_momentum, feature_downsample, expansion)

    def forward(self, representation, *args, **kwargs):

        tpv_hw, tpv_zh, tpv_wz = representation
        tpv_hw = tpv_hw.transpose(1, 2).reshape(-1, self.feature, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.transpose(1, 2).reshape(-1, self.feature, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.transpose(1, 2).reshape(-1, self.feature, self.tpv_w, self.tpv_z)
        tpv = [tpv_hw, tpv_zh, tpv_wz]
    
        x3d_l1 = tpv
        
        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l2 = self.channel_adaptor1(x3d_l2)

        x3d_l3 = self.process_l2(x3d_l2)

        x3d_l3 = self.channel_adaptor2(x3d_l3)

        outputs = [o.permute(0, 2, 3, 1).flatten(1, 2) for o in x3d_l3]

        return outputs
