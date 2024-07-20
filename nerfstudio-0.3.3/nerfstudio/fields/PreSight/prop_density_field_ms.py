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

class PropNetDensityFieldMS(nn.Module):
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
        centroids,
    ) -> None:
        super().__init__()

        self.register_buffer("centroids", deepcopy(centroids))
        self.fields = nn.ModuleList(fields)
    
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        positions = ray_samples.frustums.get_positions()
        densities = self.density_fn(positions)

        return densities, None

    def density_fn(self, positions: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 1"]:
        output_shape = positions.shape[:-1]
        positions = positions.view(-1, 3) # (bs, 3)
        bs = positions.shape[0]
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)

        densities = None
        for i, field in enumerate(self.fields):
            cluster_mask = cluster_assignments == i
            if torch.any(cluster_mask):
                _positions = positions[cluster_mask]

                _density = field.density_fn(_positions)

                if densities is None:
                    densities = torch.empty(bs, 1, dtype=_density.dtype, device=_density.device) # (bs, 1)
                densities[cluster_mask] = _density

        densities = densities.reshape(*output_shape, 1)
        return densities
    