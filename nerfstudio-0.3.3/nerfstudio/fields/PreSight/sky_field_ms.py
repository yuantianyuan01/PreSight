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
Field for sky model.
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
from nerfstudio.fields.PreSight.sky_field import SkyField
from copy import deepcopy

class SkyFieldMS(nn.Module):
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
        fields: List[SkyField],
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
        origins = ray_samples.frustums.origins[:, 0, :] # (bs, 3)

        bs = origins.shape[0]

        directions = ray_samples.frustums.directions[:, 0, :] # (bs, 3)
        if appearance_embedding is not None:
            appearance_embedding = appearance_embedding[:, 0, :] # (bs, c)
        
        cluster_assignments = torch.cdist(origins, self.centroids).argmin(dim=1)
        field_outputs = {}
        for i, field in enumerate(self.fields):
            field: SkyField
            cluster_mask = cluster_assignments == i
            if torch.any(cluster_mask):
                _directions = directions[cluster_mask]
                if appearance_embedding is not None:
                    _appearance_embedding = appearance_embedding[cluster_mask]
                else:
                    _appearance_embedding = None

                # rgb & features
                _field_outputs = field.get_outputs(_directions, _appearance_embedding)
                for k, v in _field_outputs.items():
                    if k not in field_outputs:
                        field_outputs[k] = torch.empty(bs, v.shape[-1], dtype=v.dtype, device=v.device)
                    field_outputs[k][cluster_mask] = v

        field_outputs = {k: v.contiguous() for k, v in field_outputs.items()}
        return field_outputs
