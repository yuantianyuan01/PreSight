<div align="center">
  <h1>PreSight</h1>

  <h3>[ECCV 2024] PreSight: Enhancing Autonomous Vehicle Perception with City-Scale NeRF Priors</h3>

  [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.09079)

  <img src="./resources/main_teaser_detail.jpg" width="1050px">
</div>

## Introduction

This repository is an official implementation of **PreSight: Enhancing Autonomous Vehicle Perception with City-Scale NeRF Priors**.

## News

- [2024/07/20]: :tada: We have released the code of PreSight! 

- [2024/07/09]: :confetti_ball: Our paper has been accepted by the The 18th European Conference on Computer Vision (ECCV 2024)! Our code will be release this month. Stay tuned!



## Main Results

### Online HD Mapping

|    Model     | Metric | w. Prior | Ped Crossing |  Divider  | Boundary  |           All           | Runtime (FPS) |
| :----------: | :----: | :------: | :----------: | :-------: | :-------: | :---------------------: | :-----------: |
| StreamMapNet |   AP   |    ×     |    10.19     |   11.26   |   11.87   |          11.10          |   **22.4**    |
| StreamMapNet |   AP   |    ✓     |  **21.11**   | **23.73** | **32.31** | **25.72&nbsp;(+14.62)** |     21.9      |
|    MapTR     |   AP   |    ×     |     4.97     |   8.20    |   9.83    |          7.67           |   **25.2**    |
|    MapTR     |   AP   |    ✓     |  **16.18**   | **19.04** | **34.14** | **23.12&nbsp;(+15.45)** |     23.2      |
|  BEVFormer   |  IoU   |    ×     |    14.90     |   29.88   |   32.74   |          25.84          |   **15.5**    |
|  BEVFormer   |  IoU   |    ✓     |  **16.37**   | **34.82** | **51.66** | **34.28&nbsp;(+8.44)**  |     14.3      |

### Occupancy

| Method | w. Priors |         mIoU         | Dynamic  |      **Static**       | others  | barrier  | bicycle  |   bus    |   car    | constr. vehicle | motorcycle | pedestrian | traffic cone |  truck   | drive surface | other flat | sidewalk | terrain  | manmade  | vegetation | Runtime (FPS) |
| :----: | :-------: | :------------------: | :------: | :-------------------: | :-----: | :------: | :------: | :------: | :------: | :-------------: | :--------: | :--------: | :----------: | :------: | :-----------: | :--------: | :------: | :------: | :------: | :--------: | :-----------: |
| BEVDet |     ×     |         29.3         |   24.4   |         38.2          | **1.5** | **42.4** |   11.0   | **43.0** |   47.1   |    **19.1**     |    23.3    |    23.4    |   **19.5**   | **37.8** |     72.9      |    11.6    |   30.9   |   48.6   |   32.7   |    32.5    |    **5.1**    |
| BEVDet |     ✓     | **33.7&nbsp;(+4.4)** |   24.4   | **50.5&nbsp;(+12.3)** |   1.2   |   40.1   | **14.8** |   42.1   | **48.3** |      15.7       |  **26.4**  |  **24.4**  |     18.7     |   37.2   |   **81.8**    |  **15.2**  | **40.3** | **60.5** | **50.4** |  **54.9**  |      4.9      |
| FB-Occ |     ×     |         30.0         |   25.1   |         39.2          |   9.2   |   37.2   | **21.8** | **41.6** |   43.4   |      15.8       |    27.3    |    25.4    |   **23.8**   | **30.3** |     74.7      |    17.3    |   33.0   |   50.6   |   28.2   |    31.1    |    **9.1**    |
| FB-Occ |     ✓     | **34.3&nbsp;(+4.3)** | **25.4** | **50.7&nbsp;(+11.5)** | **9.3** | **38.3** |   21.0   |   40.3   | **45.0** |    **15.9**     |  **29.9**  |  **26.0**  |     23.8     |   30.2   |   **82.3**    |  **18.5**  | **39.1** | **61.2** | **48.0** |  **54.7**  |      8.6      |



## Getting Started

To get started, please follow the instructions below step-by-step.

- [Installation](./docs/installation.md)
- [Prepare Data](./docs/prepare_data.md)
- [Building Priors](./docs/building_priors.md)
- [Training Perception Models with Priors](./docs/training_perception.md)



## Pretrained Weights

### Extracted Priors

|               |                        Boston-Seaport                        |                      Singapore-Onenorth                      |                     Singapore-Queenstown                     |                   Singapore-Hollandvillage                   |
| ------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Google  Drive | [Download](https://drive.google.com/file/d/1CtqQ6PT79v5uccKLG_bAYno64vQ3jXg0/view?usp=sharing) | [Download](https://drive.google.com/file/d/16IqKpKtje3TO_KArLYaYSh0je2vUbda3/view?usp=sharing) | [Download](https://drive.google.com/file/d/1bUBH0-qOLtgITLVAjeMgCMBZzxHwYIzN/view?usp=sharing) | [Download](https://drive.google.com/file/d/1sh95Hlmu3LlY9W8JIymtLmsLjflKSbH9/view?usp=sharing) |

### Perception Models

|               | Vectorized Online Mapping                                    | Occupancy Prediction                                         |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Google  Drive | [Download](https://drive.google.com/file/d/1_TuBJ_nqkqutS5VZpL7qTjxKqmJqH9qf/view?usp=sharing) | [Download](https://drive.google.com/file/d/1AtWqr31RYQHbzEMgfjv3HOcgv1BnoKBO/view?usp=sharing) |



## TODO

- [ ] Add scripts to inference per-image monocular depth using [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) to enable training NeRFs with monocular-depth loss. Monocular-depth loss improves visualization quality but do not improve downstream perception metrics.



### Acknowledgement

This project builds upon the outstanding work of several open-source projects. We extend our sincere thanks to the following codebases: 

- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [EmerNeRF](https://github.com/NVlabs/EmerNeRF)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [FB-BEV](https://github.com/NVlabs/FB-BEV)
- [MapTR](https://github.com/hustvl/MapTR)
- [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [SUDS](https://github.com/hturki/suds)



## Citation

If you find our work useful in your research, please consider citing:

```
@article{yuan2024presight,
  title={PreSight: Enhancing Autonomous Vehicle Perception with City-Scale NeRF Priors},
  author={Yuan, Tianyuan and Mao, Yucheng and Yang, Jiawei and Liu, Yicheng and Wang, Yue and Zhao, Hang},
  journal={arXiv preprint arXiv:2403.09079},
  year={2024}
}
```
