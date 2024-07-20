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
Put all the method implementations in one location.
"""

# Modified by Tianyuan Yuan, 2024
# Support PreSight

from __future__ import annotations

from collections import OrderedDict
from typing import Dict
from pathlib import Path
import math
import copy

import tyro
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.configs.external_methods import get_external_methods

from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.my_schedulers import WarmupMultiStepSchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.PreSight.nerfacto_nusc_ms import NerfactoNuscMSModelConfig

from nerfstudio.pipelines.PreSight.my_pipeline import MyPipelineConfig
from nerfstudio.plugins.registry import discover_methods

from nerfstudio.data.PreSight.my_datamanager import MyDataManagerConfig
from nerfstudio.data.PreSight.mynuscenes_ms_dataparser import MyNuScenesMSDataParserConfig
from nerfstudio.data.PreSight.constants import FEATURES, RGB, DEPTH, SEG, MASK

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {}

data_root = Path("path/to/your/nuScenes/")

pose_rescale_factor = 0.05
bs_scale = 8
max_iterations = 100000
for i in range(8):
    name = f"boston-seaport-monodepth-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="boston-seaport-monodepth",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="boston-seaport",
                    centroid_name=str(i),
                    num_aabbs=16,
                    use_gt_masks=False,
                    depth_type="monodepth",
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
                use_monodepth_loss=True,
                expected_depth_loss_mult=0.1,
                line_of_sight_mult=0.01,
                monodepth_depth_upperbound=25.0,
                line_of_sight_decay_steps=max_iterations,
                line_of_sight_start_step=max_iterations//20,
                line_of_sight_end_step=max_iterations,
                line_of_sight_max_sigma=6.0,
                line_of_sight_min_sigma=4.0,
                distortion_loss_mult=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

    name = f"boston-seaport-camera-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="boston-seaport-camera",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="boston-seaport",
                    centroid_name=str(i),
                    num_aabbs=16,
                    use_gt_masks=False,
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

for i in range(4):
    name = f"singapore-queenstown-monodepth-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-queenstown-monodepth",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-queenstown",
                    centroid_name="2",
                    num_aabbs=12,
                    use_gt_masks=False,
                    depth_type="monodepth",
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
                use_monodepth_loss=True,
                expected_depth_loss_mult=0.1,
                line_of_sight_mult=0.01,
                monodepth_depth_upperbound=25.0,
                line_of_sight_decay_steps=max_iterations,
                line_of_sight_start_step=max_iterations//20,
                line_of_sight_end_step=max_iterations,
                line_of_sight_max_sigma=6.0,
                line_of_sight_min_sigma=4.0,
                distortion_loss_mult=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

    name = f"singapore-queenstown-camera-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-queenstown-camera",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-queenstown",
                    centroid_name=str(i),
                    num_aabbs=12,
                    use_gt_masks=False,
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

for i in range(4):
    name = f"singapore-onenorth-monodepth-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-onenorth-monodepth",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-onenorth",
                    centroid_name=str(i),
                    num_aabbs=16,
                    use_gt_masks=False,
                    depth_type="monodepth",
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
                use_monodepth_loss=True,
                expected_depth_loss_mult=0.1,
                line_of_sight_mult=0.01,
                monodepth_depth_upperbound=25.0,
                line_of_sight_decay_steps=max_iterations,
                line_of_sight_start_step=max_iterations//20,
                line_of_sight_end_step=max_iterations,
                line_of_sight_max_sigma=6.0,
                line_of_sight_min_sigma=4.0,
                distortion_loss_mult=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

    name = f"singapore-onenorth-camera-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-onenorth-camera",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-onenorth",
                    centroid_name=str(i),
                    num_aabbs=16,
                    use_gt_masks=False,
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

for i in range(2):
    name = f"singapore-hollandvillage-monodepth-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-hollandvillage-monodepth",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-hollandvillage",
                    centroid_name=str(i),
                    num_aabbs=8,
                    use_gt_masks=False,
                    depth_type="monodepth",
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
                use_monodepth_loss=True,
                expected_depth_loss_mult=0.1,
                line_of_sight_mult=0.01,
                monodepth_depth_upperbound=25.0,
                line_of_sight_decay_steps=max_iterations,
                line_of_sight_start_step=max_iterations//20,
                line_of_sight_end_step=max_iterations,
                line_of_sight_max_sigma=6.0,
                line_of_sight_min_sigma=4.0,
                distortion_loss_mult=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

    name = f"singapore-hollandvillage-camera-dino-c{i}"
    method_configs[name] = TrainerConfig(
        output_dir=Path("./outputs"),
        experiment_name=name,
        method_name="singapore-hollandvillage-camera",
        max_num_iterations=max_iterations,
        pipeline=MyPipelineConfig(
            datamanager=MyDataManagerConfig(
                dataparser=MyNuScenesMSDataParserConfig(
                    location="singapore-hollandvillage",
                    centroid_name=str(i),
                    num_aabbs=8,
                    use_gt_masks=False,
                    data_dir=data_root,
                ),
                train_num_rays_per_batch=8192*bs_scale,
            ),
            model=NerfactoNuscMSModelConfig(
                near_plane=0.1*pose_rescale_factor,
                far_plane=1000.0*pose_rescale_factor,
                piecewise_sampler_threshold=100.0*pose_rescale_factor,
                proposal_weights_anneal_max_num_iters=max_iterations//10,
                proposal_warmup=max_iterations//10,
                use_lidar_loss=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-5),
                "scheduler": WarmupMultiStepSchedulerConfig(max_steps=max_iterations, 
                    milestones=[max_iterations//4, max_iterations//2, max_iterations*3//4], 
                    warmup_steps=max_iterations//10),
            },
        },
        vis="viewer+wandb",
    )

def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions, *discover_methods())
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
