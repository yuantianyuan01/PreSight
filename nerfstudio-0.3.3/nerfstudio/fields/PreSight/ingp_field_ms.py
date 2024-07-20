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
# Added multi-scene field

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
from nerfstudio.fields.PreSight.ingp_field import iNGPField
from copy import deepcopy

class iNGPFieldMS(nn.Module):
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
        fields: List[iNGPField],
        centroids: Tensor,
    ) -> None:
        super().__init__()

        self.register_buffer("centroids", deepcopy(centroids))
        self.fields = nn.ModuleList(fields)

    def forward(
        self, ray_samples: RaySamples, appearance_embedding: Optional[Float[Tensor, "*bs c"]]
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        positions = ray_samples.frustums.get_positions() # (*bs, 3)
        output_shape = positions.shape[:-1]

        positions = positions.reshape(-1, 3) # (bs, 3)
        bs = positions.shape[0]

        directions = ray_samples.frustums.directions.reshape(-1, 3) # (bs, 3)
        if appearance_embedding is not None:
            appearance_embedding = appearance_embedding.flatten(0, -2) # (bs, c)
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)

        field_outputs = {}
        for i, field in enumerate(self.fields):
            field: iNGPField
            cluster_mask = cluster_assignments == i
            if torch.any(cluster_mask):
                _positions = positions[cluster_mask]
                _directions = directions[cluster_mask]
                if appearance_embedding is not None:
                    _appearance_embedding = appearance_embedding[cluster_mask]
                else:
                    _appearance_embedding = None

                _density, _density_embedding = field.density_fn(_positions)

                # density
                if FieldHeadNames.DENSITY not in field_outputs:
                    field_outputs[FieldHeadNames.DENSITY] = torch.empty(bs, 1, dtype=_density.dtype, 
                                                                        device=_density.device) # (bs, 1)

                field_outputs[FieldHeadNames.DENSITY][cluster_mask] = _density

                # rgb & features
                _field_outputs = field.get_outputs(_directions, density_embedding=_density_embedding, 
                                    appearance_embedding=_appearance_embedding)
                for k, v in _field_outputs.items():
                    if k not in field_outputs:
                        field_outputs[k] = torch.empty(bs, v.shape[-1], dtype=v.dtype, device=v.device)
                    field_outputs[k][cluster_mask] = v

        field_outputs = {k: v.reshape(*output_shape, -1) for k, v in field_outputs.items()}
        return field_outputs
    
    def density_fn(self, positions: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 1"]:
        output_shape = positions.shape[:-1]
        positions = positions.view(-1, 3) # (bs, 3)
        bs = positions.shape[0]
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)

        densities = None
        density_embeddings = None
        for i, field in enumerate(self.fields):
            cluster_mask = cluster_assignments == i
            if torch.any(cluster_mask):
                _positions = positions[cluster_mask]

                _density, _density_embeddings = field.density_fn(_positions)

                if densities is None:
                    densities = torch.empty(bs, 1, dtype=_density.dtype, device=_density.device) # (bs, 1)
                    
                    density_embeddings = torch.empty(bs, _density_embeddings.shape[-1], 
                                                     dtype=_density_embeddings.dtype, 
                                                     device=_density_embeddings.device) # (bs, c)
                densities[cluster_mask] = _density
                density_embeddings[cluster_mask] = _density_embeddings

        densities = densities.reshape(*output_shape, 1)
        density_embeddings = density_embeddings.reshape(*output_shape, -1)
        return densities, density_embeddings
    
    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = ray_samples.frustums.get_positions()
        densities = self.density_fn(positions)

        return densities
    
    def semantic_fn(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*bs c"]:
        output_shape = positions.shape[:-1]
        positions = positions.view(-1, 3) # (bs, 3)
        bs = positions.shape[0]
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)

        semantics = None
        for i, field in enumerate(self.fields):
            cluster_mask = cluster_assignments == i
            if torch.any(cluster_mask):
                _positions = positions[cluster_mask]

                _semantics = field.semantic_fn(_positions)

                if semantics is None:
                    semantics = torch.empty(bs, _semantics.shape[-1], 
                                            dtype=_semantics.dtype, device=_semantics.device) # (bs, 1)
                semantics[cluster_mask] = _semantics

        semantics = semantics.reshape(*output_shape, -1)
        return semantics