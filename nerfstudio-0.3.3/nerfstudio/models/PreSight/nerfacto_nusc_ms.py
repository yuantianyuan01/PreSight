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
NeRF implementation that combines many recent advancements.
"""

# Modified by Tianyuan Yuan, 2024
# Added multi-scene field and many other improvements.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from collections import defaultdict
import functools
from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.embedding import Embedding

from nerfstudio.fields.PreSight.prop_density_field import PropNetDensityField
from nerfstudio.fields.PreSight.prop_density_field_ms import PropNetDensityFieldMS

from nerfstudio.fields.PreSight.ingp_field import iNGPField
from nerfstudio.fields.PreSight.ingp_field_ms import iNGPFieldMS
from nerfstudio.fields.PreSight.sky_field import SkyField
from nerfstudio.fields.PreSight.sky_field_ms import SkyFieldMS
from nerfstudio.fields.PreSight.utils import get_normalized_position
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.PreSight.losses import (
    line_of_sight_loss, expected_depth_loss, sky_loss, semantic_loss, z_anti_anliasing_interlevel_loss,
    expected_monodepth_loss
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, SpacedSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.PreSight.constants import SKY, DEPTH, SEG, FEATURES, RGB
import torch.nn.functional as F
from copy import deepcopy
from IPython import embed
from nerfstudio.data.PreSight.constants import VIDEO_ID

@dataclass
class NerfactoNuscMSModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoNuscMSModel)
    eval_num_rays_per_chunk: int = 1 << 15
    """specifies number of rays per chunk during eval"""
    near_plane: float = 0.1
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    num_levels: int = 10
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 16384
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 20
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 4
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (128, 64)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 1000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"features_per_level": 1, "log2_hashmap_size": 20, "num_levels": 8, 
             "base_res": 16, "max_res": 1024, "use_linear": False},
            {"features_per_level": 1, "log2_hashmap_size": 20, "num_levels": 8, 
             "base_res": 16, "max_res": 4096, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    piecewise_sampler_threshold: float = 1.0
    """Threshold for piecewise sampler"""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    enable_z_anti_aliasing: bool = True
    """Whether to enable zip-nerf interlevel loss"""
    pulse_width: Tuple[float, ...] = (0.03, 0.003)
    """z-anti-aliasing loss hyper-parameters"""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch", "tcnn+fp32"] = "tcnn+fp32"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 4
    """Dimension of the appearance embedding."""
    video_embed_dim: int = 12
    """Dimension of the video embedding."""
    use_sky_model: bool = True
    """Whether to use sky model"""
    use_ms_sky_model: bool = True
    """Whether to use sharded sky model"""
    num_sky_mlp_layers: int = 3
    """Number of mlp layers for the sky model."""
    sky_mlp_dims: int = 32
    """Number of mlp hidden dims for the sky model."""
    sky_loss_mult: float = 0.001
    """Sky loss multiplier."""
    use_lidar_loss: bool = True
    """Whether to use lidar depth model"""
    expected_depth_loss_mult: float = 1.0
    """Lidar/Monodepth depth loss multiplier"""
    lidar_depth_upperbound: float = 75.0
    """Upper bound of the depth for lidar depth loss"""
    line_of_sight_mult: float = 0.1
    """Line of sight loss multiplier"""
    line_of_sight_decay_steps: int = 5000
    """Line of sight loss decay steps"""
    line_of_sight_start_step: int = 1000
    """Line of sight loss start step"""
    line_of_sight_end_step: int = 30000
    """Line of sight loss end step"""
    line_of_sight_max_sigma: float = 5.0
    """Line of sight loss max sigma"""
    line_of_sight_min_sigma: float = 2.0
    """Line of sight loss min sigma"""

    use_semantics: bool = True
    """Whether to predict semantic features"""
    semantic_dim: int = 64
    """Semantic feature dimenstion"""
    semantic_loss_mult: float = 0.5
    """Semantic loss multiplier"""

    use_monodepth_loss: bool = False
    """Whether to use monodepth loss"""
    monodepth_loss_inverse: bool = False
    """Whether to use inverse monodepth loss"""
    monodepth_depth_upperbound: float = 40.0
    """Upper bound of the depth for monodepth loss"""



class NerfactoNuscMSModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoNuscMSModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.centroids: Float[Tensor, "nfield 3"] = self.kwargs["centroids"]
        self.aabbs: Float[Tensor, "nfield 2 3"] = self.kwargs["aabbs"]

        fields = [
            iNGPField(
                aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_semantics=self.config.use_semantics,
                semantic_dim=self.config.semantic_dim,
                appearance_embedding_dim=self.config.appearance_embed_dim + \
                    self.config.video_embed_dim,
                implementation=self.config.implementation,
            ) for aabb in self.aabbs
        ]
        self.field = iNGPFieldMS(fields, self.centroids)

        num_train_images = self.kwargs["num_train_cameras"]
        num_train_videos = self.kwargs["num_train_videos"]

        if self.config.appearance_embed_dim > 0:
            self.appearance_embedding = Embedding(num_train_images, 
                self.config.appearance_embed_dim)
        if self.config.video_embed_dim > 0:
            self.video_embedding = Embedding(num_train_videos,
                self.config.video_embed_dim)
        
        self.dino_to_rgb = self.kwargs["dino_to_rgb"]

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            prop_fields = [
                PropNetDensityField(
                    aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                ) for aabb in self.aabbs
            ]
            prop_field = PropNetDensityFieldMS(prop_fields, self.centroids)
            
            self.proposal_networks.append(prop_field)
            self.density_fns.extend([prop_field.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                prop_fields = [
                    PropNetDensityField(
                        aabb,
                        spatial_distortion=scene_contraction,
                        **prop_net_args,
                        implementation=self.config.implementation,
                    ) for aabb in self.aabbs
                ]
                prop_field = PropNetDensityFieldMS(prop_fields, self.centroids)
                self.proposal_networks.append(prop_field)
            self.density_fns.extend([prop_field.density_fn for prop_field in self.proposal_networks])

        if self.config.enable_z_anti_aliasing:
            self.interlevel_loss = functools.partial(z_anti_anliasing_interlevel_loss, 
                pulse_width=self.config.pulse_width)
        else:
            self.interlevel_loss = interlevel_loss

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        else:
            thr = self.config.piecewise_sampler_threshold
            initial_sampler = SpacedSampler(
                spacing_fn=lambda x: torch.where(x < thr, x / (2 * thr), 1 - 1 / (2 * x / thr)),
                spacing_fn_inv=lambda x: torch.where(x < 0.5, x * (2 * thr), thr / (2 - 2 * x)),
                single_jitter=self.config.use_single_jitter
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # sky model
        if self.config.use_sky_model:
            self.config.background_color = "black"
            if self.config.use_ms_sky_model:
                sky_fields = [SkyField(
                    mlp_num_layers=self.config.num_sky_mlp_layers,
                    mlp_layer_width=self.config.sky_mlp_dims,
                    appearance_embedding_dim=self.config.appearance_embed_dim + \
                        self.config.video_embed_dim,
                    use_semantics=self.config.use_semantics,
                    semantic_dim=self.config.semantic_dim,
                    implementation=self.config.implementation,
                ) for aabb in self.aabbs]

                self.sky_model = SkyFieldMS(
                    sky_fields, centroids=self.centroids
                )
            else:
                self.sky_model = SkyField(
                    mlp_num_layers=self.config.num_sky_mlp_layers,
                    mlp_layer_width=self.config.sky_mlp_dims,
                    appearance_embedding_dim=self.config.appearance_embed_dim + \
                        self.config.video_embed_dim,
                    use_semantics=self.config.use_semantics,
                    semantic_dim=self.config.semantic_dim,
                    implementation=self.config.implementation,
                )

            self.sky_loss = sky_loss
        
        if self.config.use_lidar_loss:
            assert not self.config.use_monodepth_loss
        
        if self.config.use_monodepth_loss:
            assert not self.config.use_lidar_loss

        # renderers            
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="threshold")
        self.renderer_expected_depth = DepthRenderer(method="expected")

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        if self.config.use_semantics:
            self.semantic_loss = semantic_loss

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
    
    def get_line_of_sight_sigma(self, step):
        start_step = self.config.line_of_sight_start_step
        end_step = self.config.line_of_sight_end_step
        frac = np.clip((step - start_step) / (end_step - start_step), 0., 1.)

        max_sigma = self.config.line_of_sight_max_sigma
        min_sigma = self.config.line_of_sight_min_sigma

        current_sigma = max_sigma - frac * (max_sigma - min_sigma)
        return current_sigma

    def get_line_of_sight_mult(self, step):
        if step <= self.config.line_of_sight_start_step:
            return 0.
        
        times = step // self.config.line_of_sight_decay_steps
        return self.config.line_of_sight_mult / (2. ** times)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.use_sky_model:
            param_groups["fields"] += list(self.sky_model.parameters())
        if self.config.appearance_embed_dim > 0:
            param_groups["fields"] += list(self.appearance_embedding.parameters())
        if self.config.video_embed_dim > 0:
            param_groups["fields"] += list(self.video_embedding.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)
 
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        camera_indices = ray_samples.camera_indices.squeeze(dim=-1)
        if self.training:
            if self.config.appearance_embed_dim > 0:
                assert camera_indices.max() <= self.appearance_embedding.embedding.num_embeddings - 1
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.empty((*camera_indices.shape, 0),
                    device=camera_indices.device)
            
            if self.config.video_embed_dim > 0:
                video_ids = ray_samples.metadata[VIDEO_ID].squeeze(dim=-1)
                assert video_ids.max() <= self.video_embedding.embedding.num_embeddings - 1
                embedded_video_appearance = self.video_embedding(video_ids)
                embedded_appearance = torch.cat([embedded_appearance, embedded_video_appearance], dim=-1)
        
        else:
            if self.config.use_average_appearance_embedding:
                if self.config.appearance_embed_dim > 0:
                    embedded_appearance = torch.ones(
                        (*camera_indices.shape, self.config.appearance_embed_dim), device=camera_indices.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.empty((*camera_indices.shape, 0), 
                        device=camera_indices.device
                    )
                if self.config.video_embed_dim > 0:
                    embedded_video_appearance = torch.ones(
                        (*camera_indices.shape, self.config.video_embed_dim), device=camera_indices.device
                    ) * self.video_embedding.mean(dim=0)
                    embedded_appearance = torch.cat([embedded_appearance, embedded_video_appearance], dim=-1)
                
            else:
                embedded_appearance = torch.zeros(
                    (*camera_indices.shape, self.config.appearance_embed_dim + self.config.video_embed_dim), 
                    device=camera_indices.device
                )
        
        if embedded_appearance.shape[-1] == 0:
            embedded_appearance = None
        
        field_outputs = self.field.forward(
            ray_samples, 
            appearance_embedding=embedded_appearance
        )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights) # without background blending
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        accumulation = torch.clamp(accumulation, min=0.0, max=1.0)

        sky_outputs = {}
        if self.config.use_sky_model:
            sky_outputs = self.sky_model(ray_samples, appearance_embedding=embedded_appearance)
            # +sky color
            sky_rgb = sky_outputs[FieldHeadNames.RGB]
            rgb = rgb + (1.0 - accumulation) * sky_rgb

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.use_semantics:
            semantics = field_outputs[FieldHeadNames.SEMANTICS]
            semantics = torch.sum(semantics * weights, dim=-2) # (bs, 64)
            if FieldHeadNames.SEMANTICS in sky_outputs:
                semantics = semantics + (1.0 - accumulation) * sky_outputs[FieldHeadNames.SEMANTICS]
            outputs["semantics"] = semantics
            if not self.training:
                dino_rgb = colormaps.apply_feature_colormap(semantics, self.dino_to_rgb)
                outputs["dino_rgb"] = dino_rgb
        
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["rgb"]  # RGB or RGBA image
        # gt_rgb = gt_rgb.to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        if RGB in batch:
            gt_rgb = batch[RGB]  # RGB or RGBA image
            # gt_rgb = gt_rgb.to(self.device)
            pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation( # +random background if random
                pred_image=outputs["rgb"],
                pred_accumulation=outputs["accumulation"],
                gt_image=gt_rgb,
            )
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        
        if self.config.use_sky_model and SKY in batch:
            sky_mask = batch[SKY].view(-1, 1) # 1.0 for sky
            # sky_mask = sky_mask.to(self.device)
            loss_dict["sky_loss"] = self.config.sky_loss_mult * self.sky_loss(
                outputs["accumulation"].view(-1, 1), sky_mask)
        
        if self.config.use_monodepth_loss and DEPTH in batch:
            depth = batch[DEPTH].view(-1, 1) # (bs, 1)
            ray_steps = (outputs["ray_samples_list"][-1].frustums.starts + 
                         outputs["ray_samples_list"][-1].frustums.ends) / 2

            # FIXME: workaround
            pose_scale_factor = outputs["ray_samples_list"][-1].metadata["pose_scale_factor"][0, 0, 0]
            predicted_depth = outputs["expected_depth"] / pose_scale_factor
            ray_steps = ray_steps / pose_scale_factor
            loss_dict["expected_depth_loss"] = self.config.expected_depth_loss_mult * expected_monodepth_loss(
                termination_depth=depth,
                predicted_depth=predicted_depth,
                sky_mask=sky_mask,
                upper_bound=self.config.monodepth_depth_upperbound,
                inverse=self.config.monodepth_loss_inverse,
            )

            sigma = self.get_line_of_sight_sigma(self.step)
            mult = self.get_line_of_sight_mult(self.step)
            loss_dict["line_of_sight_loss"] = mult * line_of_sight_loss(
                weights=outputs["weights_list"][-1],
                termination_depth=depth,
                steps=ray_steps,
                sigma=sigma,
                sky_mask=sky_mask,
                upper_bound=self.config.monodepth_depth_upperbound,
            )
        
        if self.config.use_lidar_loss and DEPTH in batch:
            depth = batch[DEPTH].view(-1, 1) # (bs, 1)
            # depth = depth.to(self.device)
            ray_steps = (outputs["ray_samples_list"][-1].frustums.starts + 
                         outputs["ray_samples_list"][-1].frustums.ends) / 2

            # FIXME: workaround
            pose_scale_factor = outputs["ray_samples_list"][-1].metadata["pose_scale_factor"][0, 0, 0]
            predicted_depth = outputs["expected_depth"] / pose_scale_factor
            ray_steps = ray_steps / pose_scale_factor

            loss_dict["expected_depth_loss"] = self.config.expected_depth_loss_mult * expected_depth_loss(
                termination_depth=depth,
                predicted_depth=predicted_depth,
                upper_bound=self.config.lidar_depth_upperbound,
            )
            sigma = self.get_line_of_sight_sigma(self.step)
            mult = self.get_line_of_sight_mult(self.step)
            loss_dict["line_of_sight_loss"] = mult * line_of_sight_loss(
                weights=outputs["weights_list"][-1],
                termination_depth=depth,
                steps=ray_steps,
                sigma=sigma,
                upper_bound=self.config.lidar_depth_upperbound,
            )

        if self.config.use_semantics and FEATURES in batch:
            gt_semantics = batch[FEATURES]
            # gt_semantics = gt_semantics.to(self.device)
            pred_semantics = outputs["semantics"]
            loss_dict["semantic_loss"] = self.config.semantic_loss_mult * \
                self.semantic_loss(pred=pred_semantics, target=gt_semantics, clip=True)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * self.interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            # assert metrics_dict is not None
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"])

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["rgb"]
        # gt_rgb = gt_rgb.to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_depth(self, ray_bundle: RayBundle, threshold=0.5):
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        
        density, density_embeddings = self.field.get_density(ray_samples)
        weights = ray_samples.get_weights(density)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, threshold=threshold)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        outputs = {
            "depth": depth,
            "expected_depth": expected_depth,
        }
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        return outputs
    
    @torch.no_grad()
    def get_depth_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, threshold=0.5) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        origin_shape = camera_ray_bundle.origins.shape[:-1]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.get_depth(ray_bundle=ray_bundle, threshold=threshold)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(*origin_shape, -1)  # type: ignore
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        origin_shape = camera_ray_bundle.origins.shape[:-1]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(*origin_shape, -1)  # type: ignore
        return outputs