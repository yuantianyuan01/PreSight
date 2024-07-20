# Copyright 2024 Tianyuan Yuan. All rights reserved.

import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from nuscenes.nuscenes import NuScenes
from IPython import embed
from nuscenes.utils.data_classes import LidarPointCloud
from argparse import ArgumentParser
import pickle
import cv2
from copy import deepcopy
from tqdm import tqdm

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

H, W = 900, 1600

def create_nuscenes_infos(root_path,
                          nusc=None,
                          version='v1.0-trainval',
                          scene_name="scene-0001",
                          time_threshold=0.1,
                          process_lidar=False,):
    """Create info file of nuscene dataset.

    """

    # print(f'Creating annotations of {scene_name}')
    if nusc is None:
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)

    if process_lidar:
        # read LiDARs
        timestamp_to_sweep = {}
        sample_data_token = first_sample['data']['LIDAR_TOP']
        while sample_data_token != '':
            sample_data = nusc.get('sample_data', sample_data_token)
            timestamp = float(sample_data['timestamp']) / 1e6
            filename = os.path.join(root_path, sample_data['filename'])
            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego_pose = rotation_translation_to_pose(ego_pose['rotation'], ego_pose['translation'])
            calibrated_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            sensor_pose = rotation_translation_to_pose(calibrated_sensor_data['rotation'], calibrated_sensor_data['translation'])

            pc = LidarPointCloud.from_file(filename)
            pc.remove_close(radius=1.0)

            timestamp_to_sweep[timestamp] = {
                'filename': filename,
                'ego2global': ego_pose,
                'sensor2ego': sensor_pose,
                'timestamp': timestamp,
                'point_cloud': pc
            }
            sample_data_token = sample_data['next']
        
        lidar_timestamps = np.array(sorted(list(timestamp_to_sweep.keys())))
    
    # read cameras
    sample = nusc.get('sample', first_sample_token)
    sample_data_list = []
    for camera in CAMERAS:
        first_sample_data_token = sample['data'][camera]
        sample_data_token = first_sample_data_token
        os.makedirs(os.path.join(root_path, 'lidar_depth', scene_name, camera), exist_ok=True)
        while sample_data_token != '':
            sample_data = nusc.get('sample_data', sample_data_token)
            timestamp = float(sample_data['timestamp']) / 1e6
            filename = sample_data['filename']

            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego2global_camera = rotation_translation_to_pose(ego_pose['rotation'], ego_pose['translation'])
            calibrated_sensor_data = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            
            lidar_depth_filename = os.path.join(root_path, 'lidar_depth', scene_name, sample_data['channel'], 
                                                filename.split('/')[-1].replace('jpg', 'npz'))
            if process_lidar:
                valid_indices = np.abs(timestamp - lidar_timestamps) < time_threshold
                sweeps = [timestamp_to_sweep[ts] for ts in lidar_timestamps[valid_indices]]
                sweeps = sorted(sweeps, key=lambda sweep: abs(sweep["timestamp"] - timestamp), reverse=True)
            
                depth_map = -np.ones((H, W))
                for sweep in sweeps:
                    uv, depth = sweep_to_image(sweep, calibrated_sensor_data, ego2global_camera)
                    depth_map[uv[:, 1], uv[:, 0]] = depth
            
                np.savez_compressed(lidar_depth_filename, depth_map.astype(np.float32))

            sample_data_list.append({
                'ego2global': ego2global_camera,
                'cam2ego': rotation_translation_to_pose(calibrated_sensor_data['rotation'], 
                                                        calibrated_sensor_data['translation']),
                'filename': os.path.join(root_path, filename),
                'channel': sample_data['channel'],
                'is_key_frame': sample_data['is_key_frame'],
                'height': sample_data['height'],
                'width': sample_data['width'],
                'timestamp': timestamp,
                'scene_name': scene_name,
                'cam_intrinsic': calibrated_sensor_data['camera_intrinsic'],
                # 'mask_filename': os.path.join(root_path, 'masks', scene_name, sample_data['channel'], 
                #                               filename.split('/')[-1].replace('jpg', 'png')),
                'segmentation_filename': os.path.join(root_path, 'segmentation', scene_name, sample_data['channel'], 
                                              filename.split('/')[-1].replace('jpg', 'npz')),
                'lidar_depth_filename': lidar_depth_filename,
                "dino_filename": os.path.join(root_path, "dino_features", scene_name, sample_data['channel'],
                                              filename.split('/')[-1].replace('jpg', 'npz'))
            })
            sample_data_token = sample_data['next']

    os.makedirs(os.path.join(root_path, 'PreSight'), exist_ok=True)
    output_filename = os.path.join(os.path.join(root_path, 'PreSight', scene_name + ".pkl"))
    # print(f"dumping annotations to {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(sample_data_list, f)

def rotation_translation_to_pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = Quaternion(rotation).rotation_matrix
    pose[:3, 3] = translation

    return pose

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_cam2img(pts_cam, intrinsics):
    uv = (intrinsics @ pts_cam[:3, :]).T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def sweep_to_image(sweep, camera_sample_data, ego2global_camera):
    intrinsic = np.array(camera_sample_data['camera_intrinsic'])
    extrinsic = rotation_translation_to_pose(camera_sample_data['rotation'], camera_sample_data['translation'])

    pc = deepcopy(sweep['point_cloud'])
    lidar2ego = sweep['sensor2ego']
    ego2global_lidar = sweep['ego2global']

    # transform to global
    pc.transform(lidar2ego)
    pc.transform(ego2global_lidar)

    # global to ego at camera frame
    pc.translate(-ego2global_camera[:3, 3])
    pc.rotate(ego2global_camera[:3, :3].T)

    # ego to camera
    pc.translate(-extrinsic[:3, 3])
    pc.rotate(extrinsic[:3, :3].T)

    uv, z = points_cam2img(pc.points[:3, :], intrinsic)
    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < W - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < H - 1)
    is_valid_z = z > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    uv = uv[is_valid_points]
    uv = np.round(uv).astype(np.int32)
    z = z[is_valid_points]

    points_3d = pc.points[:3, :].T
    depth = np.linalg.norm(points_3d, ord=2, axis=-1)
    depth = depth[is_valid_points]

    return uv, depth

def main(scene_names, root_path, nusc, version, process_lidar):
    pbar = tqdm(desc="Converting annotations", total=len(scene_names))
    
    for scene_name in scene_names:
        create_nuscenes_infos(root_path=root_path, nusc=nusc, version=version, scene_name=scene_name)
        pbar.update(1)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--process-lidar', action='store_true')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    scene_names = [s['name'] for s in nusc.scene]
    main(scene_names, args.data_root, nusc, args.version, args.process_lidar)
