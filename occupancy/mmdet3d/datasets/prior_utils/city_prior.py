from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree
import os
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union
import pickle
import time
from IPython import embed

class PriorPoints(object):
    def __init__(self, xyz=None, features=None, hits=None):
        self.xyz = xyz
        self.features = features
        self.hits = hits
    
    def append(self, new_xyz, new_features, new_hits):
        if self.xyz is None:
            self.xyz = new_xyz
            self.features = new_features
            self.hits = new_hits
        
        else:
            assert list(self.xyz.shape[1:]) == list(new_xyz.shape[1:])
            assert list(self.features.shape[1:]) == list(new_features.shape[1:])
            assert list(self.hits.shape[1:]) == list(new_hits.shape[1:])

            self.xyz = np.concatenate([self.xyz, new_xyz], axis=0)
            self.features = np.concatenate([self.features, new_features], axis=0)
            self.hits = np.concatenate([self.hits, new_hits], axis=0)
    
    def __repr__(self) -> str:
        r = f"xyz = {repr(self.xyz)}\n" + \
        f"features = {repr(self.features)}\n" + \
        f"hits = {repr(self.hits)}\n"

        return r
    
    def __len__(self):
        return len(self.xyz)


class NuscPrior(object):
    """NuScenes map ground-truth extractor.

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root, prior_city_parts, pc_range, prior_type="priors") -> None:
        self.pc_range = pc_range

        self.priors = {c: PriorPoints() for c in prior_city_parts}
        start = time.time()
        print("loading priors...")
        if prior_type in ["camera_priors", "monodepth_priors"]:
            for city, num_parts in prior_city_parts.items():
                for i in range(num_parts):
                    filename = os.path.join(data_root, prior_type, city, f"{city}-c{i}.pkl")
                    with open(filename, 'rb') as f:
                        p = pickle.load(f)
                    xyz = p["points"].astype(np.float32) + p["origin"].astype(np.float32)
                    xyz[:, 0:2] = - xyz[:, 0:2] # from nerfstudio coords to nuscenes coords
                    hits = p["hits"].astype(np.float32)
                    hits = hits / hits.mean()
                    self.priors[city].append(
                        new_xyz=xyz,
                        new_features=p["features"].astype(np.float16), 
                        new_hits=hits[:, None]
                    )
        
        else:
            raise ValueError(f"unknown prior type {prior_type}")
        
        self.n_dim_feats = int(self.priors[city].features.shape[-1])
        print(f"loaded priors in {time.time() - start:.2f}s")
        
    def get_prior_points(self, 
                     location: str, 
                     e2g_translation: Union[List, NDArray],
                     e2g_rotation: Union[List, NDArray]):
        ''' Extract geometries given `location` and ego pose.
        
        Args:
            location (str): city name
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global quaternion, shape (4, )
            
        Returns:
            selected_points (PriorPoints): extracted prior points.
        '''

        if location not in self.priors:
            return PriorPoints(
                xyz=np.zeros((0, 3), dtype=np.float64), 
                features=np.zeros((0, self.n_dim_feats), dtype=np.float32),
                hits=np.zeros((0, 1),dtype=np.float32)
            )

        rotation = Quaternion(e2g_rotation)
        rotation_matrix = rotation.rotation_matrix
        e2g_translation = np.array(e2g_translation)

        # construct bounding box in ego coord
        # 
        # c3 ------------- c0
        # |     ^y         |
        # |     |          |
        # |     o ---> x   |
        # c2 ------------- c1
        #
        # pc_range = [minx, miny, minz, maxx, maxy, maxz]
        ego_box_c0123 = np.array([
            [self.pc_range[3], self.pc_range[4], 0],
            [self.pc_range[3], self.pc_range[1], 0],
            [self.pc_range[0], self.pc_range[1], 0],
            [self.pc_range[0], self.pc_range[4], 0]
        ])

        global_box_c0123 = np.einsum("lk,ik->il", rotation_matrix, ego_box_c0123) + e2g_translation
        global_min_x, global_min_y, _ = global_box_c0123.min(axis=0)
        global_max_x, global_max_y, _ = global_box_c0123.max(axis=0)

        prior = self.priors[location]
        selector = (
            (prior.xyz[:, 0] <= global_max_x) &
            (prior.xyz[:, 0] >= global_min_x) &
            (prior.xyz[:, 1] <= global_max_y) &
            (prior.xyz[:, 1] >= global_min_y)
        )
        
        selected_points = PriorPoints(prior.xyz[selector].astype(np.float64), 
            prior.features[selector], prior.hits[selector])
        selected_points.xyz  = np.einsum("lk,ik->il", rotation_matrix.T, (selected_points.xyz - e2g_translation))
        selector = (
            (selected_points.xyz[:, 0] <= self.pc_range[3]) & 
            (selected_points.xyz[:, 0] >= self.pc_range[0]) & 
            (selected_points.xyz[:, 1] <= self.pc_range[4]) & 
            (selected_points.xyz[:, 1] >= self.pc_range[1]) & 
            (selected_points.xyz[:, 2] <= self.pc_range[5]) & 
            (selected_points.xyz[:, 2] >= self.pc_range[2])
        )
        selected_points = PriorPoints(selected_points.xyz[selector], selected_points.features[selector], 
            selected_points.hits[selector])

        return selected_points



