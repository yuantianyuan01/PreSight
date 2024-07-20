# Building Priors

We build the priors by first training the NeRFs and then extracting priors from them.

## Training NeRFs

We assign 8, 4, 4, and 2 tiles for Boston Seaport, Singapore Onenorth, Singapore Queenstown, and Singapore Hollandvillage, respectively. Each NeRF is trained on a single tile. To start training for a tile, simply run:

```
cd nerfstudio-0.3.3/nerfstudio/
ns-train ${location}-camera-dino-c${i_tile}
```

HHere are the commands for all tiles:

```
# 8 tiles for boston-seaport
ns-train boston-seaport-camera-dino-c0
ns-train boston-seaport-camera-dino-c1
ns-train boston-seaport-camera-dino-c2
ns-train boston-seaport-camera-dino-c3
ns-train boston-seaport-camera-dino-c4
ns-train boston-seaport-camera-dino-c5
ns-train boston-seaport-camera-dino-c6
ns-train boston-seaport-camera-dino-c7

# 4 tiles for singapore-onenorth
ns-train singapore-onenorth-camera-dino-c0
ns-train singapore-onenorth-camera-dino-c1
ns-train singapore-onenorth-camera-dino-c2
ns-train singapore-onenorth-camera-dino-c3

# 4 tiles for singapore-queenstown
ns-train singapore-queenstown-camera-dino-c0
ns-train singapore-queenstown-camera-dino-c1
ns-train singapore-queenstown-camera-dino-c2
ns-train singapore-queenstown-camera-dino-c3

# 2 tiles for singapore-hollandvillage
ns-train singapore-hollandvillage-camera-dino-c0
ns-train singapore-hollandvillage-camera-dino-c1
```

Training a single tile takes approximately 20 hours on a single A100 GPU.

**Note:** Ensure to modify the `data_root` variable [here](../nerfstudio-0.3.3/nerfstudio/configs/method_configs.py#L67) before training.



## Extracting Priors from NeRFs

After training the NeRF, we use ray marching to find surfaces and extract priors. To extract priors, run:

```
cd nerfstudio-0.3.3/nerfstudio/
python scripts/extract_priors.py ${load_dir} --downscale ${downscale} --interval ${interval}
```

where `load_dir` is the directory of the NeRF's training log, `downscale` is the image downscale factor, interval is the frame interval used for ray marching. For example, if the training log is located at `outputs/boston-seaport/boston-seaport-camera-dino-c0/2024-07-17_150059`, the command should be:

```
python scripts/extract_priors.py outputs/boston-seaport/boston-seaport-camera-dino-c0/2024-07-17_150059 --downscale 5 --interval 8
```

**Note:** The voxel-downsampling step requires substantial RAM (up to 300 GB). If you encounter OOM issues, try using a larger downscale value (e.g., 10) or a larger interval value (e.g., 12) to reduce memory requirements at the cost of priors' quality.


## Preparing the Priors for Perception Tasks

After the previous steps, the extracted priors are located in the training log's directory. Move and rename them for the downstream tasks. For example, run:

```
mv outputs/boston-seaport/boston-seaport-camera-dino-c0/2024-07-17_150059/extracted_priors.pkl /path/to/your/nuScenes/camera_priors/boston-seaport/boston-seaport-c0.pkl
```

At the end, the structure of the nuScenes directory should be:

```
nuScenes
├── camera_priors
│   ├── boston-seaport
│   │   ├── boston-seaport-c0.pkl
│   │   ├── ...
│   │   ├── boston-seaport-c7.pkl
│   ├── singapore-onenorth
│   │   ├── singapore-onenorth-c0.pkl
│   │   ├── ...
│   ├── singapore-queenstown
│   │   ├── singapore-queenstown-c0.pkl
│   │   ├── ...
│   ├── singapore-hollandvillage
│   │   ├── singapore-hollandvillage-c0.pkl
│   │   ├── ...
├── samples
├── sweeps
├── PreSight
├── v1.0-trainval
├── v1.0-test
├── ...
```

We also provide extracted priors in [google drive](https://drive.google.com/drive/folders/1qvmaH8lel0YlXIMAVbQuG2R7YKt7SNPf?usp=sharing).
