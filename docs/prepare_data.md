# Data Preparation for Builing Priors

**Important Check**: The original nuScenes camera frames require 250 GB of disk space, the extracted DINO features require 1.2 TB, and the extracted segmentation requires 20 GB. Please ensure you have sufficient disk space before processing the data.


## Preparing nuScenes

We use the full [nuScenes](https://www.nuscenes.org) dataset, including the unlabeled camera frames, for our experiments. Please download the camera frames from [here](https://www.nuscenes.org/nuscenes#download). Then modify the `data_root` variable [here](../nerfstudio-0.3.3/nerfstudio/configs/method_configs.py#L70).



## Creating Annotation Files

Create annotation files for training NeRFs:

```
cd nerfstudio-0.3.3/nerfstudio/
python scripts/datasets/create_nuscenes_infos.py --data-root path/to/nuscenes/
```

This will create a directory named `PreSight` in the data root directory and write the annotation files into it.


## Extracting Segmentation

We use [SegFormer](https://github.com/NVlabs/SegFormer) to infer segmentation results. Unfortunately it requires `pytorch<1.9`, so we need to create a new environment for it.

a. Create a new environment and activate it:

```
conda create -n segformer python=3.8 -y
conda activate segformer
```

b. Clone SegFormer in a preferred directory:

```
git clone https://github.com/NVlabs/SegFormer
```

c. Install the requirements:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf nuscenes-devkit
pip install mmcv-full==1.2.7
    
cd SegFormer
pip install .
```

d. Download the checkpoint from [the official repository](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#evaluation). We use `segformer.b5.1024x1024.city.160k.pth`. Place it under `SegFormer/pretrained/`.

d. Return to our repository and run the script. Ensure to set the `segformer-dir` to the path of your cloned SegFormer. We use multiple GPUs to accelerate the inference. Specify the `world-size` accordingly. It takes about 10 hours on 8*3090 GPUS.

```
cd nerfstudio-0.3.3/nerfstudio/
python scripts/datasets/extract_nuscenes_segmentation.py --data-root path/to/nuscenes/ --segformer-dir path/to/your/SegFormer --world-size 8 # using 8 GPUs
```


## Extracting DINO Features

a. First, we need to calculate the PCA reduction matrix. We have provided the calculated results in `scripts/datasets/pca_results.pkl` and `scripts/datasets/dino_to_rgb.pkl`, so you can skip this step if you wish. Just ensure you copy them to `path/to/nuscenes/dino_features/`.

```
cd nerfstudio-0.3.3/nerfstudio/
python scripts/datasets/extract_dino_features.py --data-root path/to/nuscenes/ --mode get_reduction_matrix --world-size 8  # using 8 GPUs
```

b. Then, infer the DINO features. We use multiple GPUs to accelerate the inference. Specify the `world-size` accordingly. It takes about 20 hours on 8*3090 GPUs.

```
python scripts/datasets/extract_dino_features.py --data-root path/to/nuscenes/ --mode get_dino --world-size 8
```
