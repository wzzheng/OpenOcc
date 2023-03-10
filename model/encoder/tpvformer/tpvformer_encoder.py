from mmseg.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer
from mmcv.runner import ModuleList
import torch.nn as nn, torch
from torch.nn.init import normal_
import copy

from ..base_encoder import BaseEncoder
from .utils import get_cross_view_ref_points, \
    get_reference_points, point_sampling
from .attention import TPVMSDeformableAttention3D, TPVCrossViewHybridAttention

@HEADS.register_module()
class TPVFormerEncoder(BaseEncoder):
    """
    Encoder used in TPVFormer, consisting of several transformer layers which 
    features cross-view hybrid-attention and image cross-attention.
    """

    def __init__(
        self,
        tpv_h=30,
        tpv_w=30,
        tpv_z=30,
        embed_dims=256,
        num_cams=6,
        num_feature_levels=4,
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        positional_encoding=None,
        num_points_in_pillar=[4, 32, 32], 
        num_points_in_pillar_cross_view=[32, 32, 32],
        transformerlayers=None, 
        num_layers=None,
        init_cfg=None):

        super().__init__(init_cfg)

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)

        # transformer layers
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.pre_norm = self.layers[0].pre_norm
        
        # other learnable embeddings
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

        # prepare reference points used in image cross-attention and cross-view hybrid-attention
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and \
            num_points_in_pillar[1] % num_points_in_pillar[0] == 0

        ref_3d_hw = get_reference_points(tpv_h, tpv_w, pc_range[5]-pc_range[2], num_points_in_pillar[0], '3d')

        ref_3d_zh = get_reference_points(tpv_z, tpv_h, pc_range[3]-pc_range[0], num_points_in_pillar[1], '3d')
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)

        ref_3d_wz = get_reference_points(tpv_w, tpv_z, pc_range[4]-pc_range[1], num_points_in_pillar[2], '3d')
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)
        
        cross_view_ref_points = get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)
        self.num_points_cross_view = num_points_in_pillar_cross_view
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or \
                isinstance(m, TPVCrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
    
    def forward_layers(
        self,
        tpv_query, # list
        key,
        value,
        tpv_pos=None, # list
        spatial_shapes=None,
        level_start_index=None,
        **kwargs
    ):
        output = tpv_query
        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)
        
        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(0).expand(
            bs, -1, -1, -1, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                tpv_h=self.tpv_h,
                tpv_w=self.tpv_w,
                tpv_z=self.tpv_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                **kwargs)
            tpv_query = output

        return output

    def forward(
        self,         
        representation,
        img_feats=None,
        metas=None,
    ):
        """Forward function.
        Args:
            img_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = img_feats[0].shape[0]
        dtype = img_feats[0].dtype
        device = img_feats[0].device

        # tpv queries and pos embeds
        tpv_queries_hw, tpv_queries_zh, tpv_queries_wz = representation
        tpv_queries_hw = tpv_queries_hw.to(dtype).unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.to(dtype).unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.to(dtype).unsqueeze(0).repeat(bs, 1, 1)

        tpv_pos_hw = self.positional_encoding(bs, device, 'z')
        tpv_pos_zh = self.positional_encoding(bs, device, 'w')
        tpv_pos_wz = self.positional_encoding(bs, device, 'h')
        tpv_pos = [tpv_pos_hw, tpv_pos_zh, tpv_pos_wz]
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # forward layers
        tpv_embed = self.forward_layers(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_pos=tpv_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=metas,
        )
        
        return tpv_embed

