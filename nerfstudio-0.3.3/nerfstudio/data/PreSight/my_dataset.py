# Copyright 2024 Tianyuan Yuan. All rights reserved.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from enum import Enum, auto
from typing import Dict, Set, List, Tuple
import functools

import torch
import numpy as np
from torch.utils.data import Dataset

from .image_metadata import ImageMetadata
from .constants import (RGB, PIXEL_INDEX, IMAGE_INDEX, RAY_INDEX, 
    VIDEO_ID, DEPTH, FEATURES,
    SKY, MASK, SEG, WIDTH)
from tqdm import tqdm
from IPython import embed
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from copy import deepcopy
import time

from .constants import CITYSCAPE_CLASSES, SKY_CLASS_ID

class ImageChunk(Dataset):
    def __init__(self,
                 rgbs,
                 segs,
                 skies,
                 depths,
                 features,
                 pixel_indices,
                 image_indices,
                 video_ids,
                 widths):
        super().__init__()
        self.rgbs = rgbs
        self.segs = segs
        self.skies = skies
        self.depths = depths
        self.features = features
        self.pixel_indices = pixel_indices
        self.image_indices = image_indices
        self.video_ids = video_ids
        self.widths = widths

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        item = {}
        item[RGB] = self.rgbs[idx]
        item[SEG] = self.segs[idx]
        item[DEPTH] = self.depths[idx]
        item[SKY] = self.skies[idx]
        
        item[IMAGE_INDEX] = self.image_indices[idx]
        item[VIDEO_ID] = self.video_ids[idx]

        pixel_index = self.pixel_indices[idx]
        width = self.widths[idx]
        
        item[RAY_INDEX] = torch.LongTensor([
            item[IMAGE_INDEX],
            pixel_index // width,
            pixel_index % width])
        
        if self.features is not None:
            item[FEATURES] = self.features[idx]

        return item

class MyDataset(Dataset):

    def __init__(self,
                 all_items: List[ImageMetadata],
                 group_flags: torch.LongTensor,
                 group_balanced: bool,
                 load_features: bool,
                 images_per_chunk: int,
                 chunk_ratio: float,
                 split: str,
                 mask_seg_classes: Tuple[str], 
                 load_on_demand: Set[str],
                 multithread: bool = False
                 ):
        super().__init__()

        self.load_features = load_features
        self.images_per_chunk = images_per_chunk
        self.chunk_ratio = chunk_ratio
        self.split = split
        self._filter_split(all_items)
        self.num_images = len(self.all_items)
        self.video_ids = torch.unique(torch.IntTensor([i.video_id for i in self.all_items]))
        self.group_flags = group_flags
        self.all_groups = torch.unique(self.group_flags)
        self.group_balanced = group_balanced

        self.num_videos = len(self.video_ids)

        self.mask_seg_classes = mask_seg_classes
        self.mask_classes_id = torch.ByteTensor([CITYSCAPE_CLASSES.index(c) 
            for c in mask_seg_classes])

        self.item_load_executor = None

        self.chunk_future = None
        self.loaded_chunk = None
        self.load_on_demand = load_on_demand
        self.multithread = multithread
    
    def load_chunk(self, modality, step) -> ImageChunk:
        while True:
            loaded_chunk = self._load_chunk_inner(modality=modality, step=step)
            if len(loaded_chunk) > 0:
                break
        
        return loaded_chunk
        
    def __len__(self) -> int:
        return len(self.all_items)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Not used now
        item = {}
        item[RGB] = self.loaded_chunk[RGB][idx]
        item[SEG] = self.loaded_chunk[SEG][idx]
        item[DEPTH] = self.loaded_chunk[DEPTH][idx]
        item[SKY] = self.loaded_chunk[SKY][idx]
        
        item[IMAGE_INDEX] = self.loaded_chunk[IMAGE_INDEX][idx]
        item[VIDEO_ID] = self.loaded_chunk[VIDEO_ID][idx]

        pixel_index = self.loaded_chunk[PIXEL_INDEX][idx]
        width = self.loaded_chunk[WIDTH][idx]
        
        item[RAY_INDEX] = torch.LongTensor([
            item[IMAGE_INDEX],
            pixel_index // width,
            pixel_index % width])
        
        if self.load_features:
            item[FEATURES] = self.loaded_chunk[FEATURES][idx]

        return item
    
    def _reset_image_choosen(self):
        self.image_choosen = torch.zeros((len(self.all_items),), dtype=torch.bool)

    def _filter_split(self, all_items):
        filtered_items = []
        for item in all_items:
            if item.is_val and self.split == "train":
                continue
            if (not item.is_val) and self.split == "val":
                continue
            filtered_items.append(item)
        
        self.all_items = filtered_items
        CONSOLE.log(f"{len(self.all_items)} images in {self.split} dataset")

    def _load_chunk_inner(self, modality: str, step: int) -> List:
        rng = np.random.default_rng(step)
        if self.images_per_chunk == -1: # load all images
            choosen_metadatas = self.all_items
        else:
            if self.group_balanced:
                choosen_metadata_idx = []
                num_image_per_group = self.images_per_chunk // len(self.all_groups)
                for i in self.all_groups:
                    group_item_idx = torch.nonzero(
                        (self.group_flags == i)
                    ).squeeze(-1)
                    choosen_idx = rng.choice(
                        group_item_idx, 
                        size=min(num_image_per_group, len(group_item_idx)), 
                        replace=False
                    ).tolist()
                    choosen_metadata_idx.extend(choosen_idx)

            else:
                choosen_metadata_idx = rng.choice(
                    np.arange(len(self.all_items)), 
                    size=min(self.images_per_chunk, len(self.all_items)), 
                    replace=False
                ).tolist()

            choosen_metadatas = [self.all_items[i] for i in choosen_metadata_idx]
        
        if modality == "image":
            fn = functools.partial(
                _load_image_chunk_by_metadatas, chunk_ratio=self.chunk_ratio, 
                load_on_demand=self.load_on_demand, mask_classes_id=self.mask_classes_id,
                load_features=self.load_features, device="cuda", multithread=self.multithread,
            )
        else:
            raise ValueError(f"unknown modality type: {modality}")

        ctx = mp.get_context("spawn")
        with mp.pool.Pool(1, context=ctx) as p:
            loaded_chunk = p.apply_async(fn, (choosen_metadatas,))
            loaded_chunk = loaded_chunk.get()
        
        # loaded_chunk = fn(choosen_metadatas)
        return loaded_chunk

    def __del__(self):
        if getattr(self, "chunk_load_executor", None):
            self.chunk_load_executor.shutdown()
        if getattr(self, "item_load_executor", None):
            if isinstance(self.item_load_executor, ThreadPoolExecutor):
                self.item_load_executor.shutdown()
            elif isinstance(self.item_load_executor, mp.pool.Pool):
                self.item_load_executor.close()

def _load_image_chunk_by_metadatas(metadatas: List[ImageMetadata], 
                                   chunk_ratio, 
                                   load_on_demand, 
                                   mask_classes_id, 
                                   load_features,
                                   multithread=False,
                                   device='cpu'):
    
    mask_classes_id = mask_classes_id.to(device)

    if multithread:
        pool = ThreadPoolExecutor(2)
        fn = functools.partial(_load_image_to_pixels,
                               chunk_ratio=chunk_ratio,
                               load_on_demand=load_on_demand,
                               mask_classes_id=mask_classes_id,
                               load_features=load_features,
                               device=device)
        results = pool.map(fn, metadatas)
        pool.shutdown()
    else:
        results = [_load_image_to_pixels(
                        m, 
                        chunk_ratio, 
                        load_on_demand,
                        mask_classes_id,
                        load_features,
                        device,
                    ) for m in metadatas
        ]

    loaded_fields = defaultdict(list)
    for pixel_dict in results:
        for k, v in pixel_dict.items():
            loaded_fields[k].append(v)

    if len(loaded_fields[RGB]) == 0:
        loaded_fields[RGB] = torch.empty((0, 3), dtype=torch.float32)
        loaded_fields[SEG] = torch.empty((0, ), dtype=torch.uint8)
        loaded_fields[DEPTH] = torch.empty((0, ), dtype=torch.float32)
        loaded_fields[PIXEL_INDEX] = torch.empty((0, ), dtype=torch.long)
        loaded_fields[IMAGE_INDEX] = torch.empty((0, ), dtype=torch.long)
        loaded_fields[VIDEO_ID] = torch.empty((0, ), dtype=torch.long)
        loaded_fields[WIDTH] = torch.empty((0, ), dtype=torch.long)
        loaded_fields[SKY] = torch.empty((0, ), dtype=torch.bool)
        loaded_fields[FEATURES] = torch.empty((0, 64), dtype=torch.float32)
    else:
        for k, v in loaded_fields.items():
            if k != FEATURES or load_features:
                loaded_fields[k] = torch.cat(v)

    loaded_chunk = ImageChunk(
        rgbs=loaded_fields[RGB], 
        segs=loaded_fields[SEG],
        skies=loaded_fields[SKY],
        depths=loaded_fields[DEPTH],
        features=loaded_fields[FEATURES] if load_features else None,
        pixel_indices=loaded_fields[PIXEL_INDEX],
        image_indices=loaded_fields[IMAGE_INDEX],
        video_ids=loaded_fields[VIDEO_ID],
        widths=loaded_fields[WIDTH]
    )
    
    if device == "cuda":
        torch.cuda.empty_cache()
    return loaded_chunk

def _load_image_to_pixels(metadata: ImageMetadata, 
                          chunk_ratio, 
                          load_on_demand, 
                          mask_classes_id, 
                          load_features, 
                          device):
    rgb = metadata.load_image(cached=(RGB not in load_on_demand), device=device).reshape(-1, 3)
    mask = metadata.load_mask(cached=(MASK not in load_on_demand), device=device).reshape(-1)
    depth = metadata.load_depth(cached=(DEPTH not in load_on_demand), device=device).reshape(-1)
    segmentation = metadata.load_segmentation(cached=(SEG not in load_on_demand), device=device).reshape(-1)

    seg_mask = ~torch.isin(segmentation, mask_classes_id)
    sky_mask = segmentation == SKY_CLASS_ID # True for sky

    image_keep_mask = torch.logical_and(mask, seg_mask)
    image_keep_indices = torch.nonzero(image_keep_mask).squeeze(-1)
    choosen_pixel_indices = torch.from_numpy(np.random.choice(
        image_keep_indices.cpu(), 
        size=int(len(image_keep_indices)*chunk_ratio), replace=False)).to(device=device)
    if load_features:
        features = metadata.load_features(cached=(FEATURES not in load_on_demand), device=device).flatten(0, -2)

    image_indices = torch.ones_like(mask, dtype=torch.long, device=device) * metadata.image_index
    video_ids = torch.ones_like(mask, dtype=torch.long, device=device) * metadata.video_id
    Ws = torch.ones_like(mask, dtype=torch.long, device=device) * metadata.W
    pixel_indices = torch.arange(metadata.W * metadata.H, dtype=torch.long, device=device)

    pixel_dict = {
        RGB: rgb[choosen_pixel_indices].cpu(),
        SEG: segmentation[choosen_pixel_indices].cpu(),
        SKY: sky_mask[choosen_pixel_indices].float().cpu(),
        DEPTH: depth[choosen_pixel_indices].cpu(),
        PIXEL_INDEX: pixel_indices[choosen_pixel_indices].cpu(),
        IMAGE_INDEX: image_indices[choosen_pixel_indices].cpu(),
        VIDEO_ID: video_ids[choosen_pixel_indices].cpu(),
        WIDTH: Ws[choosen_pixel_indices].cpu(),
        FEATURES: features[choosen_pixel_indices].cpu() if load_features else None,
        # "choosen_pixel_indices": choosen_pixel_indices,
    }

    if device == "cuda":
        del rgb, segmentation, sky_mask, depth, pixel_indices, \
        image_indices, video_ids, Ws, features

    return pixel_dict
