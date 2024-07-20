"""
Modified from https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/extract_masks.py

Modified by Tianyuan Yuan, 2024

Using SegFormer, 2021.
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf nuscenes-devkit
    pip install mmcv-full==1.2.7
    
    cd SegFormer
    pip install .

    # Download the pre-trained SegFormer model from https://github.com/NVlabs/SegFormer?tab=readme-ov-file#evaluation

Usage:
    Direct run this script in the newly set conda env.
"""


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes
from functools import partial
import multiprocessing as mp

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

SEGFORMER_DIR = "../../SegFormer"
# CONFIG_PATH = os.path.join(SEGFORMER_DIR, "local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py")
# CHECKPOINT_PATH = os.path.join(SEGFORMER_DIR, "pretrained/segformer.b5.1024x1024.city.160k.pth")

def inference_segmentation(rank, 
                           world_size, 
                           sample_data_list,
                           root_path,
                           mask_dir,
                           config_path, 
                           checkpoint_path,
                           verbose=False):

    model = init_segmentor(config_path, checkpoint_path, device=f"cuda:{rank}")
    chunksize = len(sample_data_list) // world_size + 1
    if verbose and rank == 0:
        pbar = tqdm(total=len(sample_data_list))
    for cam in CAMERAS:
        os.makedirs(os.path.join(mask_dir, cam), exist_ok=True)
    for sample_data in sample_data_list[rank*chunksize: min((rank+1)*chunksize, len(sample_data_list))]:
        img_fpath = os.path.join(root_path, sample_data['filename'])
        mask_fpath = os.path.join(mask_dir, sample_data['channel'],
            os.path.split(sample_data['filename'])[-1].replace('.jpg', '.npz'))
        result = inference_segmentor(model, img_fpath)
        mask = result[0].astype(np.uint8)
        np.savez_compressed(mask_fpath, mask)
        if verbose and rank == 0:
            pbar.update(world_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--segformer-dir', type=str, default=SEGFORMER_DIR)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    args = parser.parse_args()
    config_path = os.path.join(args.segformer_dir, "local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py")
    checkpoint_path = os.path.join(args.segformer_dir, "pretrained/segformer.b5.1024x1024.city.160k.pth")

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    all_scenes = nusc.scene
    pbar = tqdm(total=len(all_scenes),  desc="inferencing segmentation")

    for scene in all_scenes:
        scene_name = scene['name']
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        sample_data_list = []

        for camera in CAMERAS:
            first_sample_data_token = sample['data'][camera]
            sample_data_token = first_sample_data_token
            count = 0
            while sample_data_token != '':
                sample_data = nusc.get('sample_data', sample_data_token)
                sample_data_list.append(sample_data)
                sample_data_token = sample_data['next']
                count += 1
        
        # sort by timestamp (only to make chronological viz easier)
        sample_data_list.sort(key=lambda x: x["timestamp"])
        mask_dir = os.path.join(args.data_root, 'segmentation', scene_name)
        os.makedirs(mask_dir, exist_ok=True)
        fn = partial(inference_segmentation, 
                     world_size=args.world_size,
                     config_path=config_path, 
                     checkpoint_path=checkpoint_path, 
                     verbose=True,
                     sample_data_list=sample_data_list, 
                     root_path=args.data_root, 
                     mask_dir=mask_dir)

        ctx = mp.get_context('spawn')
        pool = ctx.Pool(args.world_size)
        pool.map(fn, range(args.world_size))
        pool.close()
        pool.join()
        
        pbar.update(1)
