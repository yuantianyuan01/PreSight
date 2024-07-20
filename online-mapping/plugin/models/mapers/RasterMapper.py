import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck, build_loss)

from .base_mapper import BaseMapper, MAPPERS
from IPython import embed

@MAPPERS.register_module()
class RasterMapper(BaseMapper):
    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 prior_fuse_cfg={},
                 head_cfg=dict(),
                 loss_cfg=dict(),
                 model_name=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.backbone = build_backbone(backbone_cfg)
        self.head = build_head(head_cfg)
        self.loss = build_loss(loss_cfg)

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size

        if prior_fuse_cfg:
            self.prior_fusion_module = build_neck(prior_fuse_cfg["fusion_module_cfg"])
            # self.prior_pc_range = torch.tensor(prior_fuse_cfg["pc_range"])
            # self.prior_voxel_size = torch.tensor(prior_fuse_cfg["voxel_size"])

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            self.head.init_weights()
    
    def forward_train(self, img, semantic_mask, points=None, prior_voxels=None, 
                      prior_voxels_coords=None, img_metas=None, **kwargs):
        bev_feats = self.backbone(img, img_metas=img_metas)

        # Priors
        if hasattr(self, "prior_fusion_module"):
            assert prior_voxels is not None
        if prior_voxels is not None:
            # prior_voxels = self.formulate_voxels(prior_voxels, prior_voxels_coords) # (bs, w, h, z, c)
            # prior_voxels = prior_voxels.permute(0, 4, 3, 2, 1) # (bs, c, z, h, w)
            fused_bev_feats = self.prior_fusion_module(bev_feats, prior_voxels, prior_voxels_coords)
            # fused_bev_feats = self.prior_fusion_module(bev_feats, prior_voxels)
            bev_feats = fused_bev_feats

        seg = self.head(bev_feats) # (bs, out_c, h, w)

        loss_dict = {}
        semantic_labels = semantic_mask.float() # (bs, 3, h, w)
        loss_dict["seg_loss"] = self.loss(seg, semantic_labels)
        
        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})
        num_sample = img.size(0)

        return loss, log_vars, num_sample
    
    @torch.no_grad()
    def forward_test(self, img, points=None, prior_voxels=None, 
                      prior_voxels_coords=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        bev_feats = self.backbone(img, img_metas)

        # Priors
        if hasattr(self, "prior_fusion_module"):
            assert prior_voxels is not None
        if prior_voxels is not None:
            # prior_voxels = self.formulate_voxels(prior_voxels, prior_voxels_coords) # (bs, w, h, z, c)
            # prior_voxels = prior_voxels.permute(0, 4, 3, 2, 1) # (bs, c, z, h, w)
            fused_bev_feats = self.prior_fusion_module(bev_feats, prior_voxels, prior_voxels_coords)
            bev_feats = fused_bev_feats
        
        seg = self.head(bev_feats) # (bs, put_c, h, w)

        results_list = self.head.post_process(seg, tokens)

        return results_list

    
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
            voxel = torch.zeros((*voxel_resolution, dim_feats), dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels