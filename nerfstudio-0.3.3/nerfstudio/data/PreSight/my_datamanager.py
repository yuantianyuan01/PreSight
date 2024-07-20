# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

# Modified by Tianyuan Yuan, 2024
# To support PreSight

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Literal, Set
from concurrent.futures import ThreadPoolExecutor


import numpy as np
import torch
import tyro
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.comms import get_rank, get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader
from .mynuscenes_ms_dataparser import MyNuScenesMSDataParserConfig
from .my_dataset import MyDataset
from .eval_image_dataset import EvalImageDataset
from .constants import FEATURES, RAY_INDEX, VIDEO_ID, SKY, RGB, DEPTH, SEG, MASK
from nerfstudio.utils import profiler
from time import time, sleep
from IPython import embed

CITYSCAPE_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

TIMEOUT = 300

@dataclass
class MyDataManagerConfig(DataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: MyDataManager)
    """Target class to instantiate."""
    dataparser: DataParserConfig = MyNuScenesMSDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 4096
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 8192
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval; if None, uses all val images."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(
        mode="off",
        optimizer=AdamOptimizerConfig(lr=6e-6, eps=1e-15),
        scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=250000))
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    items_per_chunk: int = 12800000 # not used for now
    """Number of entries to load into memory at a time"""
    images_per_chunk: int = 512
    """Number of images to load into memory at a time"""
    chunk_ratio: float = 0.025
    """ratio of entries used for training before loading next chunk"""
    load_on_demand: List[str] = field(default_factory=lambda: [FEATURES, RGB, DEPTH, SEG, MASK])
    """Field to load when loading a chunk. Fields not included will be cached in memory across the dataset."""
    use_group_flags: bool = False
    group_balanced: bool = True
    num_workers: int = 8
    """Number of workers for DataLoader"""
    load_features: tyro.conf.Suppress[bool] = True
    """Whether to load dino features"""
    mask_seg_classes: List[str] = field(default_factory=lambda: [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])
    """which classes to mask in training"""


class MyDataManager(DataManager):
    config: MyDataManagerConfig

    train_dataset: InputDataset  # Used by the viewer and other things in trainer
    eval_batch_dataset: MyDataset

    def __init__(
            self,
            config: MyDataManagerConfig,
            device: Union[torch.device, str] = 'cpu',
            test_mode: Literal['test', 'val', 'inference'] = 'val',
            world_size: int = 1,
            local_rank: int = 0,
            log_dir: str = "",
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = 'test' if test_mode in ['test', 'inference'] else 'val'
        self.log_dir = log_dir
        self.dataparser = self.config.dataparser.setup(log_dir=log_dir)
        
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='train', visualize=True)

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataparser_outputs.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(self.train_dataparser_outputs.cameras.to(self.device),
                                                self.train_camera_optimizer)

        self.train_batch_dataset = self._create_train_batch_dataset(self.train_dataparser_outputs)

        self.train_image_batch_dataloader = None
        self.iter_train_image_batch_dataloader = None
        self.iter_train_image_batch_dataloader_future = None

        self.dataloader_executor = ThreadPoolExecutor(2)

        # train dataset for viewer
        viewer_outputs = self.dataparser.get_dataparser_outputs(split='train', keyframe_only=True)
        self.train_dataset = InputDataset(viewer_outputs)

        # dummy, to avoid adding eval dataset in viewer
        # but we hacked the trainer to run evaluation
        self.eval_dataset = None 
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='val')
        if self.eval_dataparser_outputs is not None:
            # Construct dummy camera optimizer to make RayGenerator happy
            dummy_cam_opt = CameraOptimizerConfig(param_group="dummy_camera_opt").setup(
                num_cameras=self.eval_dataparser_outputs.cameras.size, device=self.device
            )
            self.eval_ray_generator = RayGenerator(self.eval_dataparser_outputs.cameras.to(self.device),
                                                   dummy_cam_opt)
            # batch dataset, entries are pixels
            self.eval_batch_dataset = self._create_eval_batch_dataset(self.eval_dataparser_outputs)
            start = time()
            chunk = self.eval_batch_dataset.load_chunk()
            CONSOLE.log(f"loaded eval_batch_dataset in {time() - start:.2f}s")
            self._set_eval_batch_loader(chunk)

            # image evaldataset, entries are images
            self.eval_image_dataset = EvalImageDataset(self.eval_dataparser_outputs)

        else:
            self.eval_batch_dataset = None
            self.eval_image_dataset = None

    def _create_train_batch_dataset(self, dataparser_outputs) -> MyDataset:
        return MyDataset(
            all_items=dataparser_outputs.metadata['split_items'],
            group_flags=dataparser_outputs.metadata["predicted_labels"],
            group_balanced=self.config.group_balanced,
            load_features=self.config.load_features,
            load_on_demand=set(self.config.load_on_demand),
            images_per_chunk=self.config.images_per_chunk,
            chunk_ratio=self.config.chunk_ratio,
            split="train",
            mask_seg_classes=self.config.mask_seg_classes
        )
    
    def _create_eval_batch_dataset(self, dataparser_outputs) -> MyDataset:
        return MyDataset(
            all_items=dataparser_outputs.metadata['split_items'],
            group_flags=torch.zeros((len(dataparser_outputs.metadata['split_items']),), dtype=torch.long),
            group_balanced=self.config.group_balanced,
            load_features=self.config.load_features,
            load_on_demand=set(self.config.load_on_demand),
            images_per_chunk=-1, # currently load all eval images
            chunk_ratio=1.0,
            split="val",
            mask_seg_classes=self.config.mask_seg_classes
        )

    def _get_train_batch_loader(self, modality, step):
        loaded_chunk = self.train_batch_dataset.load_chunk(modality=modality, step=step)
        if modality == "image":
            batch_size = self.config.train_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            train_sampler = DistributedSampler(loaded_chunk, self.world_size, self.local_rank)
            assert self.config.train_num_rays_per_batch % self.world_size == 0
            next_train_batch_dataloader = DataLoader(loaded_chunk, batch_size=batch_size,
                                                     sampler=train_sampler, num_workers=self.config.num_workers, 
                                                     pin_memory=False, timeout=TIMEOUT, drop_last=True)

        else:
            next_train_batch_dataloader = DataLoader(loaded_chunk, batch_size=batch_size, shuffle=True,
                                                     num_workers=self.config.num_workers, 
                                                     pin_memory=False, timeout=TIMEOUT, drop_last=True)
            
        return next_train_batch_dataloader, iter(next_train_batch_dataloader)

    def _set_train_image_batch_loader_async(self, step: int):
        """
        The motivation of this function is to pre-start the workers 
        in multi-proc iter(dataloader)
        """
        
        if self.iter_train_image_batch_dataloader_future is None:
            self.iter_train_image_batch_dataloader_future = self.dataloader_executor.submit(
                self._get_train_batch_loader, modality="image", step=step)
        # if has multi-proc iter, delete it before delete train_batch_dataloader to avoid deadlock
        if self.iter_train_image_batch_dataloader is not None:
            del self.iter_train_image_batch_dataloader
        if self.train_image_batch_dataloader is not None:
            del self.train_image_batch_dataloader
        
        (self.train_image_batch_dataloader, self.iter_train_image_batch_dataloader) = \
            self.iter_train_image_batch_dataloader_future.result()

        self.iter_train_image_batch_dataloader_future = self.dataloader_executor.submit(
            self._get_train_batch_loader, modality="image", step=step)
    
    def _set_eval_batch_loader(self, chunk):
        batch_size = self.config.eval_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(chunk, self.world_size, self.local_rank)
            assert self.config.eval_num_rays_per_batch % self.world_size == 0
            self.eval_batch_dataloader = DataLoader(chunk, batch_size=batch_size,
                                                    sampler=self.eval_sampler, num_workers=self.config.num_workers//2, 
                                                    pin_memory=False, timeout=TIMEOUT)
        else:
            self.eval_batch_dataloader = DataLoader(chunk, batch_size=batch_size, shuffle=True, 
                                                    num_workers=self.config.num_workers//2, 
                                                    pin_memory=False, timeout=TIMEOUT)

        self.iter_eval_batch_dataloader = iter(self.eval_batch_dataloader)

    def next_train_image(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1

        if self.iter_train_image_batch_dataloader is None: # self.iter_train_batch_dataloader not initialized
            batch = None
        else:
            batch = next(self.iter_train_image_batch_dataloader, None)
        while batch is None:
            self._set_train_image_batch_loader_async(step)
            batch = next(self.iter_train_image_batch_dataloader, None)
        
        batch_size = len(batch[RGB])
        if SKY in batch:
            batch[SKY] = batch[SKY].to(self.device, non_blocking=True)
        if RGB in batch:
            batch[RGB] = batch[RGB].to(self.device, non_blocking=True)
        if DEPTH in batch:
            batch[DEPTH] = batch[DEPTH].to(self.device, non_blocking=True)
        if FEATURES in batch:
            batch[FEATURES] = batch[FEATURES].to(self.device, non_blocking=True)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX].to(self.device))
        if VIDEO_ID in batch:
            ray_bundle.metadata[VIDEO_ID] = batch[VIDEO_ID].view(-1, 1).to(self.device, non_blocking=True)
        ray_bundle.metadata["pose_scale_factor"] = torch.ones(batch_size, 1) * \
            self.train_dataparser_outputs.metadata["pose_scale_factor"]

        return ray_bundle, batch
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        batch = next(self.iter_eval_batch_dataloader, None)
        if batch is None: # finished one epoch
            self._set_train_image_batch_loader_async(step)
            batch = next(self.iter_eval_batch_dataloader)
            return batch

        batch_size = len(batch[RGB])
        ray_bundle = self.eval_ray_generator(batch[RAY_INDEX])
        # ray_bundle.metadata[VIDEO_ID] = batch[VIDEO_ID].view(-1, 1).to(self.device)
        ray_bundle.metadata["pose_scale_factor"] = torch.ones(batch_size, 1) * \
            self.eval_dataparser_outputs.metadata["pose_scale_factor"]
        
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index = random.choice(range(len(self.eval_image_dataset)))
        data = self.eval_image_dataset[image_index]

        return image_index, data["ray_bundle"].to(self.device), data["batch"]
    
    def get_all_eval_images(self) -> List[Tuple[RayBundle, Dict]]:
        # TODO: multi-thread
        results = []
        for image_index in range(len(self.eval_image_dataset)):
            data = self.eval_image_dataset[image_index]
            results.append((data["ray_bundle"].to(self.device), data["batch"]))
        
        return results

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Optional[Path]:
        return Path('datapath')

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != 'off':
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups

    def __del__(self):
        if getattr(self, "dataloader_executor", None):
            self.dataloader_executor.shutdown()
