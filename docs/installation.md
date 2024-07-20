# Environment Installation Instructions

Our project involves building NeRF priors and utilizing these priors for downstream perception tasks. Each task requires different environments. Let's begin with building the NeRF priors.


## Environment for the NeRF priors

Our project is based on [Nerfstudio-0.3.3](https://github.com/nerfstudio-project/nerfstudio/tree/v0.3.3). If you encounter any issues during installation, please refer to the original repository for more details.

a. Create a new environment and activate it:

```
conda create -n presight python=3.8 -y
conda activate presight
```

b. Install pytorch. We use pytorch 1.13+cu117:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

c. Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and build the torch bindings. Please refer to the official repository for more details.

You can either install via conda and pip:

```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

or by compiling from source:

```
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j

cd bindings/torch
python setup.py install
```

d. Install our modified version of Nerfstudio:

```
cd nerfstudio-0.3.3
pip install -e .
```



## Environments for Occupancy Detection

We use [BEVDet](https://github.com/HuangJunJie2017/BEVDet) as our occupancy prediction baseline model. Below are the installation instructions. If you encounter any issues, please refer to the original repository for more details.


a. Create a new environment and activate it.

```
conda create -n bevdet python=3.8 -y
conda activate bevdet
```

b. Install pytorch. We use pytorch 1.13+cu117:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

c. Install mmcv series:

```
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.25.1 mmsegmentation==0.25.0
```

d. Install BEVDet:

```
cd occupancy
pip install -v -e .
```



## Environments for Online Mapping

We use a slightly modified version of [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) as our online mapping baseline model. Below are the installation instructions. If you encounter any issues, please refer to the original repository for more details.


a. Create a new environment and activate it:

```
conda create -n streammapnet python=3.8 -y
conda activate streammapnet
```

b. Install pytorch. We use pytorch 1.13+cu117:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

c. Install mmcv series:

```
pip install mmcv==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
pip install mmdet3d==1.0.0rc6
```

d. Install other requirements:

```
cd online-mapping
pip install -r requirements.txt
```
