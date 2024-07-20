import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, xavier_init, constant_init
import numpy as np
from mmcv.runner import BaseModule, force_fp32

@NECKS.register_module()
class PriorFusion2D(nn.Module):
    def __init__(self,
                 prior_pc_range,
                 prior_voxel_size,
                 bev_feats_channels=256, 
                 voxel_channels=68, 
                 z_pooling_size=4,
                 hidden_channels=256,
                 dropout=0.0,
                 residual=False,
                ):
        super().__init__()
        self.prior_pc_range = torch.tensor(prior_pc_range)
        self.prior_voxel_size = torch.tensor(prior_voxel_size)

        self.bev_feats_channels = bev_feats_channels
        self.voxel_channels = voxel_channels
        self.num_prior_z = int((prior_pc_range[5] - prior_pc_range[2]) / prior_voxel_size[2])
        self.z_pooling_size = z_pooling_size
        self.hidden_channels = hidden_channels
        self.num_z_pooled = int(self.num_prior_z / z_pooling_size)
        
        self.voxel_feature_extractor = nn.Sequential(
            nn.Linear(voxel_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.z_max_pooling = nn.MaxPool1d(kernel_size=self.z_pooling_size, stride=self.z_pooling_size)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels * self.num_z_pooled, hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels + bev_feats_channels, bev_feats_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(bev_feats_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(bev_feats_channels, bev_feats_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_feats_channels),
        )

        self.residual = residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        
            elif isinstance(m, nn.Linear):
                xavier_init(m)
            
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    
    @force_fp32()
    def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
        """
            bev_feats: shape (bs, bev_c, h, w)
            prior_feats: list[tensor(num_voxels, c)]
            prior_voxels_coords: list[tensor(num_voxels, 3)]
        """
        bs = len(prior_feats)
        prior_feats_list = [self.voxel_feature_extractor(p) for p in prior_feats] # list[tensor(num_voxels, hidden)]
        prior_voxels = self.formulate_voxels(prior_feats_list, prior_voxels_coords) # (bs, w, h, z, hidden)
        prior_voxels = prior_voxels.permute(0, 4, 2, 1, 3) # (bs, hidden, h, w, z)
        
        bs, hidden, h, w, z = prior_voxels.shape

        # max pooling by z-axis
        prior_feats_pooled = self.z_max_pooling(
            prior_voxels.flatten(0, -2) # (-1, num_z)
        ).view(bs, self.hidden_channels, h, w, self.num_z_pooled) # (bs, hidden, h, w, num_z_pooled)

        prior_feats_pooled = prior_feats_pooled.permute(0, 1, 4, 2, 3).flatten(1, 2) # (bs, hidden*num_z_pooled, h, w)
        
        # compress z-axis
        prior_bev_feats = self.block1(prior_feats_pooled) # (bs, hidden, h, w)

        # aggregate bev features
        if prior_bev_feats.shape[-2:] != bev_feats.shape[-2:]:
            bev_h, bev_w = bev_feats.shape[-2:]
            prior_bev_feats = F.interpolate(prior_bev_feats, size=(bev_h, bev_w),
                                            mode="bilinear", align_corners=False)
        
        x = torch.cat([bev_feats, prior_bev_feats], dim=1) # (bs, hidden+bev_c, h, w)
        if self.residual:
            bev_feats = F.relu(self.block2(x) + bev_feats) # (bs, bev_c, h, w)
        else:
            bev_feats = F.relu(self.block2(x)) # (bs, bev_c, h, w)

        return bev_feats
    
    def formulate_voxels(self, prior_voxels, prior_voxels_coords):
        bs = len(prior_voxels)
        voxel_resolution = torch.ceil(
            (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
        ).long()
        voxels = []
        for i in range(bs):
            points = prior_voxels[i]
            coords = prior_voxels_coords[i].long()
            dim_feats = points.size(1)
            voxel = torch.zeros((*voxel_resolution, dim_feats), 
                                dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels

@NECKS.register_module()
class PriorFusion2D_crossattn(nn.Module):
    def __init__(self,
                 prior_pc_range,
                 prior_voxel_size,
                 bev_feats_channels=256, 
                 voxel_channels=68, 
                 z_pooling_size=4,
                 hidden_channels=64,
                 dropout=0.0,
                 residual=False,
                 num_bev_win=10,
                 bev_h=50,
                 bev_w=100
                ):
        super().__init__()
        self.prior_pc_range = torch.tensor(prior_pc_range)
        self.prior_voxel_size = torch.tensor(prior_voxel_size)

        self.bev_feats_channels = bev_feats_channels
        self.voxel_channels = voxel_channels
        self.num_prior_z = int((prior_pc_range[5] - prior_pc_range[2]) / prior_voxel_size[2])
        self.z_pooling_size = z_pooling_size
        self.hidden_channels = hidden_channels
        self.num_z_pooled = int(self.num_prior_z / z_pooling_size)
        
        self.voxel_feature_extractor = nn.Sequential(
            nn.Linear(voxel_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.z_max_pooling = nn.MaxPool1d(kernel_size=self.z_pooling_size, stride=self.z_pooling_size)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels * self.num_z_pooled, hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, bev_feats_channels, kernel_size=1, padding=0),
        )
        
        from .window_cross_attention import WindowCrossAttention
        self.cross_attn = WindowCrossAttention(
            num_bev_win=num_bev_win,
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=bev_feats_channels,
            dropout=dropout,
        )

        self.residual = residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
        
            elif isinstance(m, nn.Linear):
                xavier_init(m)
            
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)
    
    @force_fp32()
    def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
        """
            bev_feats: shape (bs, bev_c, h, w)
            prior_feats: list[tensor(num_voxels, c)]
            prior_voxels_coords: list[tensor(num_voxels, 3)]
        """
        bs = len(prior_feats)
        prior_feats_list = [self.voxel_feature_extractor(p) for p in prior_feats] # list[tensor(num_voxels, hidden)]
        prior_voxels = self.formulate_voxels(prior_feats_list, prior_voxels_coords) # (bs, w, h, z, hidden)
        prior_voxels = prior_voxels.permute(0, 4, 2, 1, 3) # (bs, hidden, h, w, z)
        
        bs, hidden, h, w, z = prior_voxels.shape

        # max pooling by z-axis
        prior_feats_pooled = self.z_max_pooling(
            prior_voxels.flatten(0, -2) # (-1, num_z)
        ).view(bs, self.hidden_channels, h, w, self.num_z_pooled) # (bs, hidden, h, w, num_z_pooled)

        prior_feats_pooled = prior_feats_pooled.permute(0, 1, 4, 2, 3).flatten(1, 2) # (bs, hidden*num_z_pooled, h, w)
        
        # compress z-axis
        prior_bev_feats = self.block1(prior_feats_pooled) # (bs, bev_c, h, w)

        # aggregate bev features
        bev_h, bev_w = bev_feats.shape[-2:]
        if prior_bev_feats.shape[-2:] != bev_feats.shape[-2:]:
            prior_bev_feats = F.interpolate(prior_bev_feats, size=(bev_h, bev_w),
                                            mode="bilinear", align_corners=False)
        
        prior_bev_feats = prior_bev_feats.permute(0, 2, 3, 1) # (bs, h, w, bev_c)
        bev_feats = bev_feats.permute(0, 2, 3, 1) # (bs, h, w, bev_c)
        bev_feats = self.cross_attn(
            query=bev_feats.flatten(1, 2),
            key=prior_bev_feats.flatten(1, 2)).view(bs, bev_h, bev_w, self.bev_feats_channels) # (bs, h, w, bev_c)
        bev_feats = bev_feats.permute(0, 3, 1, 2) # (bs, bev_c, h, w)
        
        return bev_feats
    
    def formulate_voxels(self, prior_voxels, prior_voxels_coords):
        bs = len(prior_voxels)
        voxel_resolution = torch.ceil(
            (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
        ).long()
        voxels = []
        for i in range(bs):
            points = prior_voxels[i]
            coords = prior_voxels_coords[i].long()
            dim_feats = points.size(1)
            voxel = torch.zeros((*voxel_resolution, dim_feats), 
                                dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points.float()
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels
