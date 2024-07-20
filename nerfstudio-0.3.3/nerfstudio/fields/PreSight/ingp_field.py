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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

# Modified by Tianyuan Yuan, 2024
# Added semantic prediction

from typing import Dict, Literal, Optional, Tuple, List

import torch
from torch import Tensor, nn
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding

from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from copy import deepcopy
from .utils import get_normalized_position

class iNGPField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_semantics: whether to use semantic segmentation
        semantic_dim: dimension of semantic features
        hidden_dim_semantic_head: dimension of hidden layers for semantic feature network
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_semantics: bool = False,
        hidden_dim_semantic_head: int = 64,
        semantic_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch", "tcnn+fp32"] = "tcnn+fp32",
        field_type: Literal["iNGP", "TriPlane"] = "iNGP",
        **kwargs,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", deepcopy(aabb))
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.appearance_embedding_dim = appearance_embedding_dim
        self.use_semantics = use_semantics
        if self.use_semantics:
            self.semantic_dim = semantic_dim
        else:
            self.semantic_dim = 0
        
        self.base_res = base_res

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        if field_type == "iNGP":
            self.mlp_base_grid = HashEncoding(
                num_levels=num_levels,
                min_res=base_res,
                max_res=max_res,
                log2_hashmap_size=log2_hashmap_size,
                features_per_level=features_per_level,
                implementation=implementation,
            )
        else:
            raise ValueError(f"Unknown `field_type`: {field_type}")

        mlp_implementation = "tcnn" if implementation == "tcnn" else "torch"
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim + self.semantic_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=mlp_implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        # semantics
        if self.use_semantics:
            self.semantic_head = MLP(
                in_dim=self.semantic_dim,
                num_layers=3,
                layer_width=hidden_dim_semantic_head,
                out_dim=semantic_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=mlp_implementation,
            )

        self.rgb_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=mlp_implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        return self.density_fn(positions)

    def density_fn(self, positions: Float[Tensor, "*batch 3"], times=None) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = get_normalized_position(positions, self.aabb)
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*positions.shape[:-1], -1)
        density_before_activation, density_embedding = torch.split(h, [1, self.geo_feat_dim + self.semantic_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions.dtype))
        density = density * selector[..., None]
        return density, density_embedding

    def get_outputs(
        self, directions: Float[Tensor, "*bs 3"], 
        density_embedding: Tensor, 
        appearance_embedding: Optional[Tensor]
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        outputs_shape = directions.shape[:-1]

        # semantics
        if self.use_semantics:
            density_embedding, semantic_embedding = torch.split(
                density_embedding, 
                [self.geo_feat_dim, self.semantic_dim], 
                dim=-1
            )
            semantics_input = semantic_embedding.view(-1, self.semantic_dim)
            semantics = self.semantic_head(semantics_input).view(*outputs_shape, -1).to(density_embedding.dtype)
            outputs[FieldHeadNames.SEMANTICS] = semantics
        
        directions = get_normalized_directions(directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        if appearance_embedding is not None:
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    appearance_embedding.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                ],
                dim=-1,
            )
        
        rgb = self.rgb_head(h).view(*outputs_shape, 3).to(density_embedding.dtype)
        outputs[FieldHeadNames.RGB] = rgb

        return outputs

    def forward(self, ray_samples: RaySamples, appearance_embedding=None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples.frustums.directions, density_embedding=density_embedding, 
                                         appearance_embedding=appearance_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        return field_outputs
    
    def semantic_fn(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*bs c"]:
        assert self.use_semantics, "Cannot query semantics when `self.use_semantics` is set to False"
        
        density, density_embedding = self.density_fn(positions)
        outputs_shape = positions.shape[:-1]

        density_embedding, semantic_embedding = torch.split(
            density_embedding, 
            [self.geo_feat_dim, self.semantic_dim], 
            dim=-1
        )
        semantics_input = semantic_embedding.view(-1, self.semantic_dim)
        semantics = self.semantic_head(semantics_input).view(*outputs_shape, -1).to(density_embedding.dtype)

        return semantics