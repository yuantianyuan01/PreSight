_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_h = 480
img_w = 800
img_size = (img_h, img_w)

num_gpus = 8
batch_size = 4
num_iters_per_epoch = 24539 // (num_gpus * batch_size)
num_epochs = 24
total_iters = num_iters_per_epoch * num_epochs

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (100, 50)
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
voxel_size=[0.5, 0.5, 0.5]
prior_type = "camera_priors"

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='raster')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True

model = dict(
    type='RasterMapper',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            # pretrained='./resnet50_checkpoint.pth',
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    prior_fuse_cfg=dict(
        fusion_module_cfg=dict(
            type="PriorFusion2D",
            prior_pc_range=pc_range,
            prior_voxel_size=voxel_size,
            bev_feats_channels=bev_embed_dims,
            voxel_channels=64 + 3 + 1,
            z_pooling_size=4,
            hidden_channels=64,
            dropout=0.1,
        ),
    ),
    head_cfg=dict(
        type='BevDecoder',
        inC=bev_embed_dims,
        outC=num_class,
    ),
    loss_cfg=dict(
        type="SimpleLoss",
        pos_weight=2.13,
        loss_weight=1.0
    ),
    model_name='RasterMapper'
)

# data processing pipelines
train_pipeline = [
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        canvas_size=[400, 200],
        thickness=5,
    ),
    dict(
        type='VoxelizePriorPoints',
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels=100000,
        max_points_per_voxel=16,
        random_drop=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'semantic_mask', 'prior_voxels', 'prior_voxels_coords'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    dict(
        type='VoxelizePriorPoints',
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels=100000,
        max_points_per_voxel=16,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'prior_voxels', 'prior_voxels_coords'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='./data/nuScenes',
    ann_file='./data/nuScenes/nuscenes_map_infos_val_priorsplit.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='RasterizeMap',
            roi_size=roi_size,
            canvas_size=[400, 200],
            thickness=5,
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['semantic_mask'], meta_keys=['token'])
    ],
    interval=1,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root='./data/nuScenes',
        ann_file='./data/nuScenes/nuscenes_map_infos_train_priorsplit.pkl',
        meta=meta,
        prior_type=prior_type,
        prior_city_parts={"boston-seaport": 8, "singapore-queenstown": 4},
        prior_pc_ranges=pc_range,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-1,
    ),
    val=dict(
        type='NuscDataset',
        data_root='./data/nuScenes',
        ann_file='./data/nuScenes/nuscenes_map_infos_val_priorsplit.pkl',
        meta=meta,
        prior_type=prior_type,
        prior_city_parts={"singapore-hollandvillage": 2, "singapore-onenorth": 4},
        prior_pc_ranges=pc_range,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
    ),
    test=dict(
        type='NuscDataset',
        data_root='./data/nuScenes',
        ann_file='./data/nuScenes/nuscenes_map_infos_val_priorsplit.pkl',
        meta=meta,
        prior_type=prior_type,
        prior_city_parts={"singapore-hollandvillage": 2, "singapore-onenorth": 4},
        prior_pc_ranges=pc_range,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
    ),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
    ),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=num_epochs*num_iters_per_epoch)
find_unused_parameters = False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_epochs//6*num_iters_per_epoch, max_keep_ckpts=1)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True
