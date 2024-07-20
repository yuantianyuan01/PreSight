import torch
from torch import Tensor
from jaxtyping import Float


def get_normalized_position(positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]):
    aabb_min, aabb_max = aabb[0], aabb[1]
    positions = (positions - aabb_min) / (aabb_max - aabb_min)  # 0~1
    positions = positions * 2 - 1  # aabb is at [-1, 1]
    return positions