# Copyright 2024 Tianyuan Yuan. All rights reserved.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Dict, Set, List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from .image_metadata import ImageMetadata
from .constants import RGB
from tqdm import tqdm
from IPython import embed
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from copy import deepcopy

from .constants import CITYSCAPE_CLASSES, SKY_CLASS_ID
from .mynuscenes_ms_dataparser import DataparserOutputs

class EvalImageDataset(Dataset):

    def __init__(self,
                 dataparser_output: DataparserOutputs,
                 load_depth: bool=False,
                 load_features: bool=False,
                 load_sky: bool=False,
                 load_on_demand: Set[str]=set(),
                ):
        super().__init__()

        self.load_depth = load_depth
        self.load_features = load_features
        self.load_sky = load_sky
        self.all_items = deepcopy(dataparser_output.metadata["split_items"])
        CONSOLE.log(f"{len(self.all_items)} images in {str(self.__class__.__name__)}")

        self.cameras = deepcopy(dataparser_output.cameras)
        self.num_images = len(self.all_items)
        self.video_ids = torch.unique(torch.IntTensor([i.video_id for i in self.all_items]))
        self.num_videos = len(self.video_ids)
        self.load_on_demand = load_on_demand


    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, image_idx) -> Dict[str, torch.Tensor]:
        data = {}
        item: ImageMetadata = self.all_items[image_idx]
        rgb = item.load_image(cached=(RGB not in self.load_on_demand))
        data[RGB] = rgb
        data["image"] = deepcopy(rgb) # to be compatible with viewer
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)

        if self.load_features:
            # TODO: add features
            pass

        return {"ray_bundle": ray_bundle, "batch": data}
    