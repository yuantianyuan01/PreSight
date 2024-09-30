# Training Perception Models with Priors

After building the priors, we integrate them into downstream perception tasks. 


## Data Processing

We mainly follow the official codebases of [BEVDet](https://github.com/HuangJunJie2017/BEVDet) and [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) for data processing. We assume the nuScenes dataset is already downloaded and the priors are already extracted following [the previous section](./building_priors.md) or downloaded from [here](https://drive.google.com/drive/folders/1qvmaH8lel0YlXIMAVbQuG2R7YKt7SNPf?usp=sharing).

### For Online Mapping

a. Create softlinks:

```
cd online-mapping
mkdir data
ln -s /path/to/your/nuScenes/ ./data/nuScenes
```

b. Create annotations for our proposed new data split:

```
conda activate streammapnet # remember to switch environments
python tools/data_converter/nuscenes_converter.py --data-root ./data/nuScenes
```


### For Occupancy

a. Create softlinks:

```
cd occupancy
mkdir data
ln -s /path/to/your/nuScenes/ ./data/nuscenes
```

b. Create annotations for our proposed new data split:

```
conda activate bevdet # remember to switch environments
python tools/create_data_bevdet.py --data-root ./data/nuscenes --extra_tag priorsplit
```


## Training Models

### For Online Mapping

#### Vectorized

a. To training a model with or without priors on the new data split:

```
cd online-mapping
conda activate streammapnet # remember to switch environments

# model with priors
bash tools/dist_train.sh ./plugin/configs/smn_wcamprior_480_100x50_24e_randomdrop.py 8 # 8-GPU training

# model without priors
bash tools/dist_train.sh ./plugin/configs/smn_priorsplit_480_100x50_24e.py 8 # 8-GPU training
```

b. To test a checkpoint:

```
cd online-mapping
conda activate streammapnet # remember to switch environments

# model with priors
bash tools/dist_test.sh ./plugin/configs/smn_wcamprior_480_100x50_24e_randomdrop.py /path/to/checkpoint 8 # 8-GPU testing

# model without priors
bash tools/dist_test.sh ./plugin/configs/smn_priorsplit_480_100x50_24e.py /path/to/checkpoint 8 # 8-GPU testing
```

#### Rasterized

Replace the config file by [nusc_raster_wcamprior_480_100x50_24e_randomdrop.py](../online-mapping/plugin/configs/nusc_raster_wcamprior_480_100x50_24e_randomdrop.py) and [nusc_raster_priorsplit_480_100x50_24e](../online-mapping/plugin/configs/nusc_raster_priorsplit_480_100x50_24e).

### For Occupancy

a. To training a model with priors on the new data split:

```
cd occupancy
conda activate bevdet # remember to switch environments

# model with priors
bash tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-r50d-8x4-24e_wcamprior_randomdrop.py 8 # 8-GPU training

# model without priors
bash tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-r50d-8x4-24e_priorsplit.py 8 # 8-GPU training
```

**Note:** Following the official codebase of [BEVDet](https://github.com/HuangJunJie2017/BEVDet), training this model requires a pre-trained backbone, which can be downloaded [here](https://drive.google.com/file/d/1xDeQLdHmGts6ULa75PuUhEA0Y_KoB5YU/view?usp=sharing) or from the official codebase.

b. To training a baseline model without priors on the new data split:

```
cd occupancy
conda activate bevdet # remember to switch environments

# model with priors
bash tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-r50d-8x4-24e_wcamprior_randomdrop.py /path/to/checkpoint 8 # 8-GPU testing

# model with priors
bash tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-r50d-8x4-24e_priorsplit.py /path/to/checkpoint 8 # 8-GPU testing
```

**Note: ** In BEVDet, the evaluation results of the last training epoch do not use the EMA checkpoint, which may result in a marginal performance drop. We recommend running the test again using the EMA checkpoint to get the best results.
