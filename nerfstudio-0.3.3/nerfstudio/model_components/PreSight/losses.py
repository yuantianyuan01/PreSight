from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift
import torch.nn.functional as F
from nerfstudio.model_components.losses import ray_samples_to_sdist

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0
DEPTH_METRIC = 1

def normalize_depth(depth: Tensor, upper_bound: float = 75.0):
    return torch.clip(depth / upper_bound, 0.0, 1.0)

def line_of_sight_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
    sky_mask: Optional[Float[Tensor, "*batch 1"]] = None, # 1.0 for sky
    upper_bound: float = 75.0,
) -> Float[Tensor, "*batch 1"]:
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        steps: Sampling distances along rays.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = (termination_depth > 1.) & (termination_depth < upper_bound)
    if sky_mask is not None:
        depth_mask = depth_mask & (sky_mask == 0.0)
    
    steps = steps.detach()
    # Line of sight losses
    termination_depth = termination_depth[:, None]
    target_distribution = torch.distributions.normal.Normal(0.0, sigma / URF_SIGMA_SCALE_FACTOR)
    line_of_sight_loss_near_mask = torch.logical_and(
        steps <= termination_depth + sigma, steps >= termination_depth - sigma
    )
    line_of_sight_loss_near = (weights - torch.exp(target_distribution.log_prob(steps - termination_depth))) ** 2
    line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)
    line_of_sight_loss_empty_mask = steps < termination_depth - sigma
    line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * weights**2).sum(-2)
    line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty

    # loss = line_of_sight_loss * depth_mask
    return torch.mean(line_of_sight_loss[depth_mask])

def expected_depth_loss(
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    upper_bound: float = 75.0
) -> Float[Tensor, "*batch 1"]:
    
    depth_mask = (termination_depth > 1.) & (termination_depth < upper_bound)

    termination_depth = normalize_depth(termination_depth, upper_bound=upper_bound)
    predicted_depth = normalize_depth(predicted_depth, upper_bound=upper_bound)

    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2
    # loss = expected_depth_loss * depth_mask
    return torch.mean(expected_depth_loss[depth_mask])

def expected_monodepth_loss(
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    sky_mask: Float[Tensor, "*batch 1"], # 1.0 for sky[Tensor, "*batch 1"]
    upper_bound: float = 50.0,
    inverse: bool = False
) -> Float[Tensor, "*batch 1"]:
    
    depth_mask = (termination_depth > 1.) & (termination_depth < upper_bound) & (sky_mask == 0.0)

    if inverse:
        termination_depth = 1 / (termination_depth + 5)
        predicted_depth = 1 / (predicted_depth + 5)
    else:
        termination_depth = normalize_depth(termination_depth, upper_bound=upper_bound)
        predicted_depth = normalize_depth(predicted_depth, upper_bound=upper_bound)

    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2
    # loss = expected_depth_loss * depth_mask
    return torch.mean(expected_depth_loss[depth_mask])


def sky_loss(
    accumulation: Float[Tensor, "*batch 1"], 
    sky_mask: Float[Tensor, "*batch 1"], # 1.0 for sky
    eps: float = 1e-7):

    target = 1.0 - sky_mask # 0.0 for sky
    accumulation = torch.clip(accumulation, min=eps, max=1-eps)
    # loss = -(target * prob.log() + (1 - target) * (1 - prob).log())
    loss = F.binary_cross_entropy(accumulation, target, reduction='none')
    return loss.mean()

def semantic_loss(
    pred: Float[Tensor, "*batch c"],
    target: Float[Tensor, "*batch c"],
    clip: bool = True
):
    if clip:
        # pred = torch.clip(pred, min=0.0, max=1.0)
        target = torch.clip(target, min=0.0, max=1.0)
    return F.mse_loss(pred, target, reduction="none").mean()

def blur_stepfun(x, y, r):
    # taken and modified from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/stepfun.py
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (
        torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1)
        - torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
    ) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum(
        (xr[..., 1:] - xr[..., :-1]) * torch.cumsum(y2, dim=-1), dim=-1
    ).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr

def sorted_interp_quad(x, xp, fpdf, fcdf):
    # taken and modified from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/stepfun.py
    """interp in quadratic"""
    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret

def z_anti_anliasing_interlevel_loss(weights_list, ray_samples_list, pulse_width):
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach() # (num_rays, num_samples + 1)
    w = weights_list[-1][..., 0].detach() # (num_rays, num_samples)
    assert len(ray_samples_list) > 0

    w_normalized = w / (c[..., 1:] - c[..., :-1])
    c1, w1 = blur_stepfun(c, w_normalized, pulse_width[0])
    c2, w2 = blur_stepfun(c, w_normalized, pulse_width[1])

    area1 = 0.5 * (w1[..., 1:] + w1[..., :-1]) * (c1[..., 1:] - c1[..., :-1])
    area2 = 0.5 * (w2[..., 1:] + w2[..., :-1]) * (c2[..., 1:] - c2[..., :-1])
    cdfs1 = torch.cat(
        [
            torch.zeros_like(area1[..., :1]),
            torch.cumsum(area1, dim=-1),
        ],
        dim=-1,
    )
    cdfs2 = torch.cat(
        [
            torch.zeros_like(area2[..., :1]),
            torch.cumsum(area2, dim=-1),
        ],
        dim=-1,
    )
    cs = [c1, c2]
    ws = [w1, w2]
    _cdfs = [cdfs1, cdfs2]

    loss_interlevel = 0.0
    for i, (ray_samples, weights) in enumerate(zip(ray_samples_list[:-1], weights_list[:-1])):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        cdf_interp = sorted_interp_quad(
            cp, cs[i], ws[i], _cdfs[i]
        )
        w_s = torch.diff(cdf_interp, dim=-1)
        loss_interlevel += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
    
    return loss_interlevel