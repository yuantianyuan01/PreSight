import numpy as np
import mmcv
import numba
import os
import cv2
import torch

from mmdet.datasets.builder import PIPELINES
from numpy import random
from IPython import embed

@PIPELINES.register_module(force=True)
class VoxelizePriorPoints(object):
    """
    
    """
    def __init__(self, pc_range, voxel_size, max_voxels=20000, max_points_per_voxel=35, 
            load_features=True, random_drop=False, max_drop_rate=1.0, pose_error_scale=0.0):
        self.pc_range = np.array(pc_range)
        self.voxel_size = np.array(voxel_size)
        self.load_features = load_features
        self.random_drop = random_drop
        self.max_drop_rate = max_drop_rate
        self.pose_error_scale = pose_error_scale

        assert np.all(np.ceil((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size) == 
                np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)), \
                f"pc_range {pc_range} must be divided by voxel_size {voxel_size}!"

        self.voxel_resolution = np.ceil(
            (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        ).astype(np.int64)
        self.max_voxels = max_voxels
        self.max_points_per_voxel = max_points_per_voxel
    
    @staticmethod
    def world_coords_to_voxel_coords(
        points: np.ndarray,
        pc_range: np.ndarray,
        voxel_resolution: np.ndarray,
    ) -> np.ndarray:
        """
        Convert points in world coordinates to voxel coordinates.

        Args:
            point (ndarray): The points to convert.
            pc_range (ndarray): The bounding box, [minx, miny, minz, maxx, maxy, maxz]
            voxel_resolution (ndarray): The number of voxels in each dimension of the voxel grid.

        Returns:
            Tensor: The voxel coordinates of the points.
        """
        # Convert lists to tensors if necessary
        aabb_min = pc_range[:3]
        aabb_max = pc_range[3:]
        # Compute the size of each voxel
        voxel_size = (aabb_max - aabb_min) / voxel_resolution
        # Compute the voxel index for the given point
        voxel_coords = ((points - aabb_min) / voxel_size).astype(np.int64)
        return voxel_coords
    
    def __call__(self, results: dict):
        points = results["prior_points"]
        if self.load_features:
            new_points = np.concatenate([
                points.xyz.astype(np.float64), 
                points.features.astype(np.float64),
                points.hits.astype(np.float64)
            ], axis=-1)
        else:
            new_points = np.concatenate([
                points.xyz.astype(np.float64), 
                points.hits.astype(np.float64)
            ], axis=-1)

        ################## to simulate random pose error ##################
        if self.pose_error_scale > 0:
            noise_xyz = np.random.normal(scale=self.pose_error_scale)
            new_points[:, :3] += noise_xyz
        ###################################################################

        valid_flag = (new_points[:, 0] >= self.pc_range[0]) & (new_points[:, 0] <= self.pc_range[3]) & \
                     (new_points[:, 1] >= self.pc_range[1]) & (new_points[:, 1] <= self.pc_range[4]) & \
                     (new_points[:, 2] >= self.pc_range[2]) & (new_points[:, 2] <= self.pc_range[5])
        new_points = new_points[valid_flag]

        if len(new_points) == 0:
            results.update({
                "prior_voxels": new_points.astype(np.float32), # (0, c)
                "prior_voxels_coords": np.zeros((0, 3), dtype=np.int32), # (0, 3)
            })
            return results
        
        ################## to fit with BEVDet's BEV data augmentation ##################
        rotate_bda = results.get('rotate_bda', 0)
        flip_dx = results.get('flip_dx', False)
        flip_dy = results.get('flip_dy', False)
        scale_ratio = results.get('scale_bda', 1.0)

        rotate_angle = rotate_bda / 180 * np.pi
        rot_sin = np.sin(rotate_angle)
        rot_cos = np.cos(rotate_angle)
        scale_mat = np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        flip_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ np.array([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ np.array([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        new_points[:, :3] = np.einsum("ik,jk->ji", rot_mat, new_points[:, :3])
        #################################################################################

        np.random.shuffle(new_points)
        voxels, coords, num_points_per_voxel = points_to_voxel(
            points=np.ascontiguousarray(new_points),
            voxel_size=self.voxel_size,
            coors_range=self.pc_range,
            max_voxels=self.max_voxels,
            max_points=self.max_points_per_voxel
        )
        # average features and positions by hit count
        hit_weighted_sum_feats = (voxels[:, :, :-1] * voxels[:, :, -1:]).sum(axis=1)
        hit_weighted_average_feats  = hit_weighted_sum_feats / voxels[:, :, -1:].sum(axis=1)
        hit_sum = voxels[:, :, -1:].sum(axis=1)
        voxels = np.concatenate([hit_weighted_average_feats, hit_sum], axis=-1) # (num, c)
        
        assert not (np.any(np.isnan(voxels)) or np.any(np.isinf(voxels))), "nan or inf in voxels!"

        # normalize xyz and hit
        range_xyz = self.pc_range[3:] - self.pc_range[:3]
        voxels[:, :3] = (voxels[:, :3] - self.pc_range[:3]) / range_xyz # to (0, 1)
        assert voxels[:, -1:].min() > 0.
        voxels[:, -1:] = np.log(voxels[:, -1:])

        if self.random_drop:
            keep_rate = 1 - np.random.uniform(0, self.max_drop_rate)
            keep_idx = np.random.choice(np.arange(len(voxels)), 
                                        size=int(keep_rate * len(voxels)),
                                        replace=False)
            voxels = voxels[keep_idx]
            coords = coords[keep_idx]
        
        results.update({
            "prior_voxels": voxels.astype(np.float32), # (num, c)
            "prior_voxels_coords": coords.astype(np.int32), # (num, 3)
        })
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


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
        for i in range(N):
            f.write(f'{points[i, 0]:.3f} {points[i, 1]:.3f} {points[i, 2]:.3f} {c[i, 0]} {c[i, 1]} {c[i, 2]}\n')

def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function creates.
            For second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                        num_points_per_voxel,
                                        coor_to_voxelidx, voxels, coors,
                                        max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = grid_size.astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num
