# Copyright 2024 Tianyuan Yuan. All rights reserved.

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CITYSCAPE_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

class ImageMetadata:
    def __init__(self, image_path: str, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, 
                 image_index: int, time: str, video_id: int, is_val: bool=False, is_key_frame: bool=False,
                 depth_path: Optional[str]=None, 
                 mask_path: Optional[str]=None, seg_path: Optional[str]=None, sky_mask_path: Optional[str]=None,
                 feature_path: Optional[str]=None, local_cache: Optional[Path]=None):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self.time = time
        self.video_id = video_id
        self.is_key_frame = is_key_frame
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.seg_path = seg_path
        self.sky_mask_path = sky_mask_path
        self.feature_path = feature_path
        self.is_val = is_val
        self.cached_pixel_dict = {}

    def load_image(self, cached=False, device='cpu') -> torch.Tensor:
        if cached and getattr(self, "rgb_cache", None) is not None:
            return self.rgb_cache

        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        rgb = torch.tensor(np.array(rgbs), dtype=torch.float32, device=device) / 255.

        if cached:
            self.rgb_cache = rgb
        
        return rgb

    def load_mask(self, cached=False, device='cpu') -> torch.Tensor:
        if self.mask_path is None:
            mask = torch.ones(self.H, self.W, dtype=torch.bool, device=device)
            if "CAM_BACK" in self.image_path and (
                "CAM_BACK_RIGHT" not in self.image_path and "CAM_BACK_LEFT" not in self.image_path):
                # If is backcam, mask the truck of the ego vehicle
                truck_height = int(self.H / 9)
                mask[-truck_height:] = False
            return mask
        
        if cached and getattr(self, "mask_cache", None) is not None:
            return self.mask_cache

        mask = np.array(Image.open(self.mask_path))
        mask = torch.tensor(mask, dtype=torch.uint8, device=device)
        size = mask.shape

        if size[0] != self.H or size[1] != self.W:
            mask = mask.reshape(1, 1, size[0], size[1])
            mask = F.interpolate(mask, size=(self.H, self.W), mode='nearest-exact')
            mask = mask.reshape(self.H, self.W)

        mask = mask > 0
        if "CAM_BACK" in self.image_path and (
            "CAM_BACK_RIGHT" not in self.image_path and "CAM_BACK_LEFT" not in self.image_path):
            # If is backcam, mask the truck of the ego vehicle
            truck_height = int(self.H / 9)
            mask[-truck_height:] = False
        if cached:
            self.mask_cache = mask
        
        return mask

    def load_depth(self, cached=False, device='cpu') -> torch.Tensor:
        if self.depth_path is None:
            return -torch.ones(self.H, self.W, dtype=torch.float32, device=device)

        if cached and getattr(self, "depth_cache", None) is not None:
            return self.depth_cache

        depth = np.load(self.depth_path, mmap_mode='r')
        
        if isinstance(depth, np.lib.npyio.NpzFile):
            depth = depth['arr_0']
        
        depth = torch.tensor(depth, dtype=torch.float32, device=device)
        size = depth.shape

        if size[0] != self.H or size[1] != self.W:
            depth = depth.reshape(1, 1, size[0], size[1])
            depth = F.interpolate(depth, size=(self.H, self.W), mode='nearest-exact')
            depth = depth.reshape(self.H, self.W)

        if cached:
            self.depth_cache = depth
        
        return depth
    
    def load_segmentation(self, cached=False, device='cpu') -> torch.Tensor:
        if self.seg_path is None:
            return torch.zeros(self.H, self.W, dtype=torch.bool, device=device)

        if cached and getattr(self, "seg_cache", None) is not None:
            return self.seg_cache
        
        seg = np.load(self.seg_path, mmap_mode='r')
        
        if isinstance(seg, np.lib.npyio.NpzFile):
            seg = seg['arr_0']
        
        seg = torch.tensor(seg, dtype=torch.uint8, device=device)
        size = seg.shape

        if size[0] != self.H or size[1] != self.W:
            seg = seg.reshape(1, 1, size[0], size[1])
            seg = F.interpolate(seg, size=(self.H, self.W), mode='nearest-exact')
            seg = seg.reshape(self.H, self.W)
        
        if cached:
            self.seg_cache = seg

        return seg
    
    def load_features(self, cached=False, device='cpu') -> torch.Tensor:
        if self.feature_path is None:
            return torch.zeros(self.H, self.W, dtype=torch.bool, device=device)

        if cached and getattr(self, "feature_cache", None) is not None:
            return self.feature_cache
        
        features = np.load(self.feature_path, mmap_mode='r')
        
        if isinstance(features, np.lib.npyio.NpzFile):
            features = features['arr_0']
        
        features = torch.tensor(features, dtype=torch.float32, device=device)
        size = features.shape # (h, w, c)

        if size[0] != self.H or size[1] != self.W:
            features = features.unsqueeze(0).permute(0, 3, 1, 2) # (1, c, h, w)
            features = F.interpolate(features, size=(self.H, self.W), mode='nearest-exact') # (1, c, H, W)
            features = features.permute(0, 2, 3, 1).squeeze() # (H, W, c)
        
        if cached:
            self.feature_cache = features

        return features
