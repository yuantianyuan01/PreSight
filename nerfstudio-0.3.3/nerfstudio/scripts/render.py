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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import os
import struct
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from collections import defaultdict
import mediapy as media
import numpy as np
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.PreSight.image_metadata import ImageMetadata
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.pipelines.PreSight.my_pipeline import MyPipeline

from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from IPython import embed
from PIL import Image
from tqdm import tqdm
from nerfstudio.data.PreSight.constants import CITYSCAPE_CLASSES, SKY_CLASS_ID

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

def _render_trajectory_video(
    pipeline: MyPipeline,
    cameras: Cameras,
    imagemetas: List[ImageMetadata],
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    fps: float = 24,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
    """
    CONSOLE.log("Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    image_height, image_width = int(cameras.height[0]), int(cameras.width[0])
    for item in imagemetas:
        item.H = image_height
        item.W = image_width
    cameras = cameras.to(pipeline.device)
    pose_scale_factor = pipeline.datamanager.dataparser.config.pose_scale_factor
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        colored_frames = defaultdict(list)
        for camera_idx in tqdm(range(cameras.size), desc="rendering video", dynamic_ncols=True):
            aabb_box = None
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
            
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            for rendered_output_name in rendered_output_names:
                if rendered_output_name == "depth":
                    output_image = colormaps.apply_depth_colormap(
                            outputs[rendered_output_name] / pose_scale_factor,
                            # accumulation=outputs["accumulation"],
                            near_plane=depth_near_plane,
                            far_plane=depth_far_plane,
                            colormap_options=colormap_options,
                        ).cpu().numpy()
                
                elif rendered_output_name == "accumulation":
                    output_image = colormaps.apply_depth_colormap(
                            outputs[rendered_output_name],
                            accumulation=None,
                            near_plane=0.0,
                            far_plane=1.0,
                            colormap_options=colormap_options,
                        ).cpu().numpy()
                
                elif rendered_output_name == "mask":
                    dynamic_mask = ~imagemetas[camera_idx].load_mask().numpy()
                    output_image = imagemetas[camera_idx].load_image().numpy()
                    output_image[dynamic_mask, :] = output_image[dynamic_mask, :] * 0.5
                    
                    segmentation = imagemetas[camera_idx].load_segmentation().numpy()
                    mask_classes_id = np.array([CITYSCAPE_CLASSES.index(c) for c in ["sky"]], dtype=np.uint8)
                    seg_mask = np.isin(segmentation, mask_classes_id)
                    output_image[seg_mask, :] = 0

                elif rendered_output_name == "gt_rgb":
                    output_image = imagemetas[camera_idx].load_image().numpy()

                elif rendered_output_name == "rgb":
                    output_image = colormaps.apply_colormap(
                            image=outputs[rendered_output_name],
                            colormap_options=colormap_options,
                        ).cpu().numpy()
                    
                elif rendered_output_name == "dino_rgb":
                    output_image = outputs[rendered_output_name].cpu().numpy()
                
                elif rendered_output_name == "dino_gt":
                    output_image = imagemetas[camera_idx].load_features()
                    dino_to_rgb = pipeline.model.dino_to_rgb
                    output_image = colormaps.apply_feature_colormap(output_image, dino_to_rgb).numpy()

                else:
                    raise ValueError(f"Invalid rendered_output_name: {rendered_output_name}")
                    
                colored_frames[rendered_output_name].append(output_image)

        stacked_frames = []
        for frame_idx in range(len(cameras) // len(CAMERAS)):
            frame_hstack = []
            for key in rendered_output_names:
                six_views = colored_frames[key][
                    frame_idx*len(CAMERAS): (frame_idx+1)*len(CAMERAS)]
                six_views = np.hstack(six_views)
                frame_hstack.append(six_views)
            stacked_frames.append(np.vstack(frame_hstack))
        
        render_height = int(stacked_frames[0].shape[0])
        render_width = int(stacked_frames[0].shape[1])
        writer = stack.enter_context(
            media.VideoWriter(
                path=output_filename,
                shape=(render_height, render_width),
                fps=fps,
            )
        )
        if output_format == "video":
            for frame in stacked_frames:
                writer.add_image(frame)

    CONSOLE.log(f"Render Complete: {str(output_filename)}")


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_dir: Path
    """Path to config YAML file."""
    downscale: float = 4.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: int = 1 << 16
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: 
        ["rgb", "gt_rgb", "depth", "dino_gt", "dino_rgb", "mask", "accumulation"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: float = 0.0
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: float = 60.0
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""

@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    def main(self) -> None:
        """Main function."""
        output_path = self.load_dir / "output.mp4"
        config = self.load_dir / "config.yml"
        _, pipeline, _, _ = eval_setup(
            config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()
        # if self.pose_source == "eval":
        #     assert pipeline.datamanager.eval_dataset is not None
        #     cameras = pipeline.datamanager.eval_dataset.cameras
        # else:
        #     assert pipeline.datamanager.train_dataset is not None
        #     cameras = pipeline.datamanager.train_dataset.cameras
        
        cameras = {}
        for k in CAMERAS:
            cameras[k] = pipeline.datamanager.dataparser.get_dataparser_outputs(split='all', 
                channel=k).metadata['split_items']
        
        front_timestamps = [c.time for c in cameras["CAM_FRONT"]]
        camera_list = []
        for ts in front_timestamps:
            for k in CAMERAS:
                timestamps = np.array([c.time for c in cameras[k]])
                nearest_idx = np.argmin(np.abs(timestamps - ts))
                camera_list.append(cameras[k][nearest_idx])
        # camera_list = camera_list[:100]
        cameras = pipeline.datamanager.dataparser.create_cameras(camera_list)
        fps = 12

        _render_trajectory_video(
            pipeline,
            cameras,
            imagemetas=camera_list,
            output_filename=output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale,
            fps=fps,
            output_format="video",
            image_format="jpeg",
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


# Commands = tyro.conf.FlagConversionOff[
#     Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
# ]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderInterpolated).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(RenderInterpolated)  # noqa
