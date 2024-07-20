# Copyright 2024 Tianyuan Yuan. All rights reserved.

from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
import numpy as np
import torch
import pickle
import time

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.PreSight.nerfacto_nusc_ms import NerfactoNuscMSModel
from nerfstudio.pipelines.PreSight.my_pipeline import MyPipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.colormaps import apply_feature_colormap
from IPython import embed
from tqdm import tqdm
from nerfstudio.data.PreSight.constants import CITYSCAPE_CLASSES, SKY_CLASS_ID
from argparse import ArgumentParser

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
NUM_CAMERAS = 6

eval_num_rays_per_chunk = 1 << 17
colormap_options = colormaps.ColormapOptions()
depth_near_plane: float = 0.0
depth_far_plane: float = 60.0

@torch.no_grad()
def extract_voxels(
    pipeline: MyPipeline,
    output_dir: str,
    device: str = "cuda",
    frame_interval: int = 1,
    camera_scaling_factor: float = 1.0,
    voxel_size: float = 0.4,
    max_depth: float = 50.0,
    min_depth: float = 0.5,
    hit_thr_ratio: float = 0.2,
    depth_type: str = "depth",
    use_segmentation_mask: bool = True,
):
    '''
    Extract featured points from the model and downsample them to voxels.
    Results are saved in a pickle file and visualized in a ply file.

    Args:
        pipeline: The model's pipeline object.
        output_dir: The directory to save the output.
        device: default "cuda"
        frame_interval: The number of frames to skip for each camera.
        camera_scaling_factor: The factor to scale the camera resolution by.
        voxel_size: The size of the voxels to downsample to.
        max_depth: The maximum depth to do ray marching.
        min_depth: The minimum depth to do ray marching.
        hit_thr_ratio: How many percent of the points to keep.
        depth_type: The type of depth to use, either "depth" or "expected_depth".
        use_segmentation_mask: Whether to use the segmentation mask to filter out invalid pixels.
    '''

    model: NerfactoNuscMSModel = pipeline.model
    model.eval()
    train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    dino_to_rgb = train_dataparser_outputs.metadata["dino_to_rgb"]
    pose_scale_factor = train_dataparser_outputs.metadata["pose_scale_factor"]
    origin = train_dataparser_outputs.metadata["pose_tranformation"]
    image_metas = train_dataparser_outputs.metadata["split_items"]
    for i in image_metas:
        H = int(i.H * camera_scaling_factor)
        W = int(i.W * camera_scaling_factor)
        scale_matrix = torch.tensor([
            [W / i.W, 0,  0],
            [0, H / i.H, 0],
            [0, 0, 1]
            ], dtype=torch.float32)
        i.H = H
        i.W = W
        i.intrinsics = scale_matrix @ i.intrinsics
    mask_classes_id = pipeline.datamanager.train_batch_dataset.mask_classes_id.to(device)

    cameras: Cameras = train_dataparser_outputs.cameras.to(device)
    cameras.rescale_output_resolution(camera_scaling_factor)
    cameras_coords = cameras.get_image_coords().to(device)
    camera_opt = pipeline.datamanager.train_camera_optimizer
    
    frame_indices = range(0, len(cameras) // NUM_CAMERAS + 1, frame_interval)
    camera_indices = list(range(len(cameras)))
    camera_indices = reduce(lambda x, y: x + y, 
                        [camera_indices[NUM_CAMERAS*i: NUM_CAMERAS*(i+1)] for i in frame_indices])
    
    all_hit_points = []
    all_hit_points_features = []
    all_hit_points_densities = []
    all_hit_points_colors = []
    for i, camera_idx in enumerate(tqdm(camera_indices, desc="inferencing", dynamic_ncols=True)):
        if use_segmentation_mask:
            metadata = image_metas[camera_idx]
            seg = metadata.load_segmentation(device=device)
            seg_mask = ~torch.isin(seg, mask_classes_id) # True for valid pixels
            coords = torch.nonzero(seg_mask)
            coords = cameras_coords[coords[:, 0], coords[:, 1]] # same as coords + 0.5
        else:
            coords = cameras_coords

        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, coords=coords, aabb_box=None)
        if len(camera_ray_bundle.origins) == 0:
            continue
        outputs = model.get_depth_for_camera_ray_bundle(camera_ray_bundle)

        depth = outputs[depth_type] / pose_scale_factor
        world_coords = (
            camera_ray_bundle.origins / pose_scale_factor + camera_ray_bundle.directions * depth
        ).view(-1, 3)
        depth = depth.flatten()
        selector = (
            (depth < max_depth)
            & (depth > min_depth)
            & (world_coords[:, 2] > -3.0)
            & (world_coords[:, 2] < 6.0)
        )

        world_coords = world_coords[selector] # (n, 3)
        N = len(world_coords)
        if N == 0: 
            continue
        density_list = []
        # we need to accumulate the density from proposal networks as well
        # to ensure reliable density estimation
        for p in model.proposal_networks:
            density_list.append(p.density_fn(world_coords * pose_scale_factor).squeeze(-1))
        results = model.field.density_fn(world_coords * pose_scale_factor)[0].squeeze(-1)
        density_list.append(results)
        densities_mean = torch.stack(density_list, dim=0).mean(dim=0) # (n,)
        feats = model.field.semantic_fn(world_coords * pose_scale_factor).clip(0., 1.).to(torch.float16) # (n, 64)
        pca_colors = apply_feature_colormap(feats, dino_to_rgb) # (n, 64)
        
        assert len(pca_colors) == N and len(feats) == N and len(densities_mean) == N
        all_hit_points.append(world_coords.cpu())
        all_hit_points_colors.append(pca_colors.cpu())
        all_hit_points_densities.append(densities_mean.cpu())
        all_hit_points_features.append(feats.cpu())
        del world_coords, pca_colors, densities_mean, feats
        
        if i % 100 == 0:
            torch.cuda.empty_cache()

    all_hit_points = torch.cat(all_hit_points)
    all_hit_points_colors = torch.cat(all_hit_points_colors)
    all_hit_points_densities = torch.cat(all_hit_points_densities)
    all_hit_points_features = torch.cat(all_hit_points_features)

    print(f"num hit points before density thr: {len(all_hit_points)}")
    selector = all_hit_points_densities > 1.0
    print(f"num hit points after density thr: {int(selector.sum())}")
    all_hit_points_thr = all_hit_points[selector].numpy()
    del all_hit_points
    all_hit_points_colors_thr = all_hit_points_colors[selector].numpy()
    del all_hit_points_colors
    all_hit_points_features_thr = all_hit_points_features[selector].numpy()
    del all_hit_points_features
    
    start = time.time()
    ds_points, ds_colors, ds_indices = points_downsample_to_voxels(
        points=all_hit_points_thr,
        voxel_size=voxel_size,
        colors=None,
    )
    print(f"finished downsampling in {time.time() - start:.2f}s")
    print(f"num voxels after downsampled to {voxel_size}: {len(ds_points)}")

    colors = []
    features = []
    hits = []
    for i, points_indices in enumerate(tqdm(ds_indices, desc="tracing features", dynamic_ncols=True)):
        indices = np.asarray(points_indices)
        color = all_hit_points_colors_thr[indices].mean(axis=0)
        feature = all_hit_points_features_thr[indices].astype(np.float64).mean(axis=0).astype(np.float16)

        hits.append(len(indices))
        colors.append(color)
        features.append(feature)
    
    hits = np.asarray(hits)
    colors = np.stack(colors)
    features = np.stack(features)
    hit_thr = np.quantile(np.asarray(hits), hit_thr_ratio)
    selector = hits > hit_thr

    points_thr = ds_points[selector]
    colors_thr = colors[selector]
    features_thr = features[selector]
    print(f"num voxels before hit thr: {len(ds_points)}")
    print(f"num voxels after hit thr: {len(points_thr)}")

    output_path = os.path.join(output_dir, f"extracted_priors.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            "points": points_thr.astype(np.float32), 
            "features": features_thr.astype(np.float16), 
            "colors": colors_thr.astype(np.float32), 
            "hits": hits[selector],
            "origin": origin.numpy().astype(np.float32)
        }, f)
    print(f"result saved to {output_path}")
    
    output_path = os.path.join(output_dir, f"priors_for_vis.ply")
    
    write_ply(points_thr, colors_thr, output_path)
    print(f"ply saved to {output_path}")

def points_downsample_to_voxels(points, voxel_size, colors=None):
    '''
    Downsample points to voxels.

    Args:
        points (np.ndarray): Points in shape (N, 3).
        voxel_size (float): The size of the voxels.
        colors (np.ndarray, optional): Colors of the points in shape (N, 3). Defaults to None.
    
    Returns:
        The downsampled points, colors, and indices of the voxels.
    '''
    import open3d as o3d

    print("converting to vector3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    min_bound = points.min(axis=0).reshape(3, 1) - 1.0
    max_bound = points.max(axis=0).reshape(3, 1) + 1.0

    print("computing downsampling")
    new_pcd, _, indices = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)
    
    if colors is not None:
        colors = np.array(new_pcd.colors)

    return np.array(new_pcd.points), colors, indices


def write_ply(points, colors, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    with open(out_filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uint8 red\n"
                "property uint8 green\n"
                "property uint8 blue\n"
                "end_header\n")
        
        c = (colors * 255).astype(np.uint8)
        for i in tqdm(range(N), desc="writing file", dynamic_ncols=True):
            f.write(f'{points[i, 0]:.3f} {points[i, 1]:.3f} {points[i, 2]:.3f} {c[i, 0]} {c[i, 1]} {c[i, 2]}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("load_dir", type=str, help="load dir")
    parser.add_argument("--downscale", type=float, default=5.0, 
                        help="downscale factor for image, balance with interval")
    parser.add_argument("--interval", type=int, default=8,
                        help="number of frames to skip for each camera, balance with downscale factor")
    parser.add_argument("--hit-ratio", type=float, default=0.0,
                        help="ratio of points to keep, between 0 and 1")
    parser.add_argument("--voxel-size", type=float, default=0.4,
                        help="voxel size for downsampling")

    args = parser.parse_args()
    """Main function."""
    config = os.path.join(args.load_dir, "config.yml")
    _, pipeline, _, _ = eval_setup(
        Path(config).absolute(),
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        test_mode="test",
    )
    
    extract_voxels(
        pipeline=pipeline,
        output_dir=args.load_dir,
        camera_scaling_factor=1.0 / args.downscale,
        frame_interval=args.interval,
        device="cuda",
        voxel_size=args.voxel_size,
        max_depth=50.0,
        min_depth=0.5,
        hit_thr_ratio=args.hit_ratio,
        depth_type="expected_depth",
    )

