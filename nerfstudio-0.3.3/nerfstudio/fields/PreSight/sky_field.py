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
# Added semantic field

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions

class SkyField(nn.Module):
    """Sky Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        direction_encoding: str = "SHEncoding",
        mlp_num_layers: int = 3,
        mlp_layer_width: int = 64,
        appearance_embedding_dim: int = 32,
        use_semantics: bool = False,
        semantic_dim: int = 64,
        implementation: Literal["tcnn", "torch", "tcnn+fp32"] = "tcnn+fp32"
    ) -> None:
        super().__init__()
        
        self.use_semantics = use_semantics
        self.appearance_embedding_dim = appearance_embedding_dim
        if direction_encoding == 'SHEncoding':
            self.direction_encoding = SHEncoding(
                levels=4, implementation=implementation
            )

        mlp_implementation = "tcnn" if implementation == "tcnn" else "torch"
        self.rgb_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.appearance_embedding_dim,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=mlp_implementation,
        )
        if self.use_semantics:
            self.semantic_head = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=mlp_num_layers,
                layer_width=mlp_layer_width,
                out_dim=semantic_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=mlp_implementation,
            )
    
    def get_outputs(self, directions: Float[Tensor, "bs 3"], 
                    appearance_embedding: Optional[Float[Tensor, "bs c"]]):
        normalized_directions = get_normalized_directions(directions)
        d = self.direction_encoding(normalized_directions)

        outputs = {}
        if appearance_embedding is not None:
            sky_rgb = self.rgb_head(torch.cat([d, appearance_embedding], dim=-1)) # (bs, 3)
        else:
            sky_rgb = self.rgb_head(d) # (bs, 3)
        outputs[FieldHeadNames.RGB] = sky_rgb

        if self.use_semantics:
            sky_semantics = self.semantic_head(d)
            outputs[FieldHeadNames.SEMANTICS] = sky_semantics # (bs, c)
        
        return outputs
    
    def forward(
        self, ray_samples: RaySamples, appearance_embedding: Optional[Float[Tensor, "*bs c"]]
    ) -> Dict[FieldHeadNames, Tensor]:
        directions = ray_samples.frustums.directions[:, 0, :].contiguous() # (bs, 3)
        if appearance_embedding is not None:
            appearance_embedding = appearance_embedding[:, 0, :].contiguous() # (bs, c)
        
        return self.get_outputs(directions, appearance_embedding)


