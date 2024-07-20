# Copyright 2024 Tianyuan Yuan. All rights reserved.

IMAGE_INDEX = 'image_index'
PIXEL_INDEX = 'pixel_index'
RGB = 'rgb'
DEPTH = 'depth'
FEATURES = 'features'

RAY_INDEX = 'ray_index'
WIDTH = 'width'
TIME = 'time'
VIDEO_ID = 'video_id'

MASK = 'mask'
SEG = 'seg'

SKY = 'sky'

CITYSCAPE_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

SKY_CLASS_ID = CITYSCAPE_CLASSES.index("sky")

