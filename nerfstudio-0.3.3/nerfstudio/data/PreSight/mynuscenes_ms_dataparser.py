# Copyright 2024 Tianyuan Yuan. All rights reserved.

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type, List

import numpy as np
import pyquaternion
import torch
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
import pickle
import json

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE
from IPython import embed
from .image_metadata import ImageMetadata
from sklearn.cluster import KMeans
from copy import deepcopy

# To make sklearn kmeans happy
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose



@dataclass
class MyNuScenesMSDataParserConfig(DataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use nerfstudio/scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: MyNuScenesMSDataParser)
    """target class to instantiate"""
    # data: Path = Path("scene-0001")  # TODO: rename to scene but keep checkpoint saving name?
    """Name of the scene."""
    scene_names: Optional[List[str]] = None
    """Name of the scenes."""
    centroid_name: str = "0"
    """Name of centroid"""
    location: str = "singapore-onenorth"
    """Name of location"""
    data_dir: Path = Path("../../data/nuScenes")
    """Path to NuScenes dataset."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-trainval"
    """Dataset version."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"], ...] = (
        "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT")
    """Which cameras to use."""
    train_split_fraction: float = 1.0
    """The percent of images to use for training. The remaining images are for eval."""
    num_aabbs: int = 1
    """Num aabbs"""
    image_downscale_factor: float = 1.0
    """Image downscale factor, should be in (0.0, 1.0]"""
    pose_scale_factor: float = 0.05
    """pose downscale factor, should be in (0.0, 1.0]"""
    pose_normalize: bool = True
    """Whether to normalize pose to 0 mean"""
    use_gt_masks: bool = False
    """Whether to use ground truth dynamic object masks instead of prediction from segmentation model."""
    depth_type: Literal["lidar", "monodepth", "none"] = "none"
    """Which depth type to use."""


@dataclass
class MyNuScenesMSDataParser(DataParser):
    """NuScenes DatasetParser"""

    config: MyNuScenesMSDataParserConfig

    def __init__(self, config: DataParserConfig, log_dir: str = ""):
        super().__init__(config)
        self.log_dir = log_dir

    def _generate_dataparser_outputs(self, 
                                     split="train", 
                                     keyframe_only=False, 
                                     channel="all",
                                     visualize=False):
        transform1 = torch.tensor(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )
        transform2 = torch.tensor(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )
        if keyframe_only:
            CONSOLE.log(f'parsing nuScenes split {split} keyframe...')
        else:
            CONSOLE.log(f'parsing nuScenes split {split}...')
        
        if channel == "all":
            cameras = ["CAM_" + camera for camera in self.config.cameras]
        else:
            cameras = [channel]

        sample_data_list = []

        with open(os.path.join("./scripts/datasets/",
            f"{self.config.location}",
            f"{self.config.location}_centroids.json"), 'r') as f:
            self.scene_names = json.load(f)[self.config.centroid_name]
        
        for scene_name in self.scene_names:
            with open(os.path.join(str(self.config.data_dir), "PreSight", f'{scene_name}.pkl'), 'rb') as f:
                sample_data_list.extend(pickle.load(f))

        dino_dir = os.path.join(str(self.config.data_dir), "dino_features")
        if not os.path.exists(dino_dir):
            dino_dir = os.path.join(str(self.config.data_dir), "dino_features_fp16")

        with open(os.path.join(dino_dir, "dino_to_rgb.pkl"), 'rb') as f:
            dino_to_rgb = pickle.load(f)
            dino_to_rgb = {k: torch.tensor(v, dtype=torch.float32) for k, v in dino_to_rgb.items()}

        # sort by timestamp
        sample_data_list.sort(key=lambda x: x["timestamp"])

        # get image filenames and camera data
        image_filenames = []
        all_items = []
        data_dir = str(self.config.data_dir.resolve())
        for i, sample_data in enumerate(sample_data_list):
            if sample_data['channel'] not in cameras:
                continue

            ego_pose = torch.tensor(sample_data['ego2global'], dtype=torch.float32)
            cam_pose = torch.tensor(sample_data['cam2ego'], dtype=torch.float32)
            pose = ego_pose @ cam_pose

            # rotate to opencv frame
            pose = transform1 @ pose

            # convert from opencv camera to nerfstudio camera
            pose[0:3, 1:3] *= -1
            pose = pose[np.array([1, 0, 2, 3]), :]
            pose[2, :] *= -1

            # rotate to z-up in viewer
            pose = transform2 @ pose
            img_fpath = sample_data['filename']
            image_filenames.append(img_fpath)
            
            mask_fpath = sample_data.get("mask_filename", None)
            seg_fpath = sample_data['segmentation_filename']
            depth_fpath = sample_data.get("lidar_depth_filename", None)
            if self.config.depth_type == "monodepth":
                depth_fpath = depth_fpath.replace("lidar_depth", "monodepth")
            feature_path = sample_data['dino_filename']
            if dino_dir.split("/")[-1] == "dino_features_fp16":
                feature_path = feature_path.replace("dino_features", "dino_features_fp16")

            H = int(sample_data['height'] * self.config.image_downscale_factor)
            W = int(sample_data['width'] * self.config.image_downscale_factor)

            scale_matrix = torch.tensor([
                [W / sample_data['width'], 0,  0],
                [0, H / sample_data['height'], 0],
                [0, 0, 1]
                ], dtype=torch.float32)
            intrinsic = scale_matrix @ torch.tensor(sample_data['cam_intrinsic'], dtype=torch.float32)

            item = ImageMetadata(
                image_path=img_fpath,
                c2w=pose, 
                W=W, 
                H=H, 
                intrinsics=intrinsic,
                image_index=i,
                time=sample_data["timestamp"], 
                video_id=self.scene_names.index(sample_data["scene_name"]),
                is_key_frame=sample_data["is_key_frame"],
                mask_path=mask_fpath if self.config.use_gt_masks else None, 
                seg_path=seg_fpath,
                depth_path=depth_fpath if self.config.depth_type != "none" else None,
                feature_path=feature_path,
                is_val=False,
            )
            all_items.append(item)
        
        poses = torch.stack([item.c2w for item in all_items])
        translations = poses[:, :3, 3]
        
        # dummy scene_box for vis
        x_min, y_min, z_min = translations.min(0)[0]
        x_max, y_max, z_max = translations.max(0)[0]
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[x_min, y_min, -10], [x_max, y_max, -25]],
                dtype=torch.float32
            )
        )

        if split == "train" and not keyframe_only:
            # k-means to find #num_aabbs cluster
            kmeans = KMeans(n_clusters=self.config.num_aabbs, 
                            random_state=0, 
                            n_init="auto",
                            max_iter=500).fit(translations.numpy())
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

            predicted_distance = kmeans.transform(translations.numpy()).min(-1)
            CONSOLE.log(f"median = {np.median(predicted_distance):.2f}, "
                        f"mean = {np.mean(predicted_distance):.2f}, "
                        f"max = {np.max(predicted_distance):.2f}, "
                        f"min = {np.min(predicted_distance):.2f}"
            )
            predicted_labels = torch.tensor(kmeans.predict(translations.numpy()), dtype=torch.long)
            predicted_distance = torch.from_numpy(predicted_distance)
            poses_by_centroids = {}
            dist_by_centroids = {}
            for i in range(self.config.num_aabbs):
                p = translations[predicted_labels == i]
                poses_by_centroids[i] = p
                dist_by_centroids[i] = torch.norm(p - centroids[i], dim=-1)

            aabbs = []
            log_aabb_sizes = []
            for i, centroid in enumerate(centroids):
                # aabb_size = torch.quantile(dist_by_centroids[i], 0.98) + 20
                # log_aabb_size.append(round(aabb_size.item(), 2))
                # aabbs.append(torch.stack([
                #     centroid - aabb_size,
                #     centroid + aabb_size,
                # ]))
                p = poses_by_centroids[i]
                c_x, c_y, c_z = centroid
                x_max, y_max, z_max = torch.quantile(p, 0.95, dim=0)
                x_min, y_min, z_min = torch.quantile(p, 0.05, dim=0)
                
                aabb = torch.tensor([
                    [x_min - 15, y_min - 15, z_min - 5],
                    [x_max + 15, y_max + 15, z_max + 15]
                ], dtype=torch.float32)
                
                aabbs.append(aabb)
                log_aabb_sizes.append(aabb[1] - aabb[0])
            CONSOLE.log(f"aabbs: {log_aabb_sizes}")
            if visualize:
                self.visualize_centroids(all_items, centroids, aabbs)
            aabbs = torch.stack(aabbs)
        
        else:
            aabbs = torch.zeros((self.config.num_aabbs, 2, 3))
            centroids = torch.zeros((self.config.num_aabbs, 3))
            predicted_labels = None

        # pose_normalize
        if self.config.pose_normalize:
            _mean = translations.mean(dim=-2)
            poses[:, :3, 3] -= _mean
            aabbs = aabbs - _mean
            centroids = centroids - _mean
            scene_box.aabb -= _mean
        else:
            _mean = torch.zeros((1, 3), dtype=torch.float32)
        
        # NOTE: pose_scale_factor workaround
        pose_scale_factor = self.config.pose_scale_factor
        poses[:, :3, 3] = poses[:, :3, 3] * pose_scale_factor
        for i, item in enumerate(all_items):
            item.c2w = poses[i, :].contiguous()
        aabbs = aabbs * pose_scale_factor
        centroids = centroids * pose_scale_factor
        scene_box.aabb *= pose_scale_factor
        
        # filter key-frames after pose normalization
        if keyframe_only:
            all_items = [item for item in all_items if item.is_key_frame]

        # filter image_filenames and poses based on train/eval split percentage
        num_snapshots = len(all_items)
        num_train_snapshots = math.ceil(num_snapshots * self.config.train_split_fraction)
        num_eval_snapshots = num_snapshots - num_train_snapshots
        i_all = np.arange(num_snapshots)
        i_train = np.linspace(
            0, num_snapshots - 1, num_train_snapshots, dtype=int
        )  # equally spaced training snapshots starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_snapshots

        if split == "train":
            split_indices = i_train
        elif split in ["val", "test"]:
            split_indices = i_eval
        elif split == "all":
            split_indices = i_all
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        
        # set correct image indices for each split
        train_count, val_count = 0, 0
        for i, item in enumerate(all_items):
            if i in i_eval:
                item.is_val = True
                item.image_index = val_count
                val_count += 1
            else:
                item.is_val = False
                item.image_index = train_count
                train_count += 1
        
        split_items = [item for i, item in enumerate(all_items) if i in split_indices]
        image_filenames = [filename for i, filename in enumerate(image_filenames) if i in split_indices]

        if len(split_items) == 0:
            return None
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=self.create_cameras(split_items),
            scene_box=scene_box,
            mask_filenames=None,
            metadata={
                'split_items': split_items,
                # 'all_cameras': self.create_cameras(all_items),
                "pose_scale_factor": pose_scale_factor,
                "pose_tranformation": _mean,
                "dino_to_rgb": dino_to_rgb,
                "centroids": centroids,
                "aabbs": aabbs,
                "predicted_labels": predicted_labels,
            }
        )
        return dataparser_outputs
    
    @staticmethod
    def create_cameras(metadata_items: List[ImageMetadata]) -> Cameras:
        return Cameras(
            camera_to_worlds=torch.stack([x.c2w[:3, :4] for x in metadata_items]),
            fx=torch.FloatTensor([x.intrinsics[0, 0] for x in metadata_items]),
            fy=torch.FloatTensor([x.intrinsics[1, 1] for x in metadata_items]),
            cx=torch.FloatTensor([x.intrinsics[0, 2] for x in metadata_items]),
            cy=torch.FloatTensor([x.intrinsics[1, 2] for x in metadata_items]),
            width=torch.IntTensor([x.W for x in metadata_items]),
            height=torch.IntTensor([x.H for x in metadata_items]),
            camera_type=CameraType.PERSPECTIVE,
        )

    def visualize_centroids(self, all_items, centroids, aabbs):
        from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
        import matplotlib.pyplot as plt
        from shapely.geometry import box

        nusc_map = NuScenesMap(dataroot=str(self.config.data_dir), map_name=self.config.location)
        rng = np.random.default_rng(0) # use new rng to prevent affecting randomness
        fig, ax = nusc_map.render_layers(['drivable_area'])
        
        scene_list = []
        for i, item in enumerate(all_items):
            scene_list.append(deepcopy(item.c2w[:3, 3].tolist()))
            if i == len(all_items) - 1 or (all_items[i + 1].video_id != item.video_id):
                # end of one scene
                _poses = np.array(scene_list)[:, :2]
                color = list(rng.random(size=3))
                # nerfstudio coordinates to nuScenes
                ax.plot(-_poses[:, 0], -_poses[:, 1], 'o-', color=color, alpha=0.3, linewidth=2, markersize=2)
                scene_list = []
        
        # nerfstudio coordinates to nuScenes
        ax.plot(-centroids[:, 0].numpy(), -centroids[:, 1].numpy(), 'x', 
                color='r', markersize=8, alpha=1.0)
        
        for aabb in aabbs:
            bounding_box = box(aabb[0][0], aabb[0][1], aabb[1][0], aabb[1][1])
            ax.plot(
                -np.array(bounding_box.exterior.coords)[:, 0], 
                -np.array(bounding_box.exterior.coords)[:, 1], 
                '-', color='r', linewidth=1
            )
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        
        fpath = os.path.join(self.log_dir, "kmeans.png")
        CONSOLE.log(f"Visualization in {fpath}")
        try:
            plt.savefig(fpath, dpi=300)
        except PermissionError as e:
            CONSOLE.log(f"{str(e)}\nskip visualization")
