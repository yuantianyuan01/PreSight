'''
Modified from https://github.com/NVlabs/EmerNeRF/blob/main/third_party/feature_extractor.py
Modified by Tianyuan Yuan, 2024
'''
import math
import os
import types
from typing import Literal, Optional, Tuple, Type, List, Union
from torch import Tensor
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch.nn.modules.utils as nn_utils
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from IPython import embed
import pickle
from time import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import random
from functools import partial
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes
from sklearn.decomposition import PCA
from time import sleep

# from utils.visualization_tools import get_robust_pca, to8b

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

H, W = 900, 1600
INPUT_IMAGE_SHAPE = (576, 1024)
STRIDE = 8
MODEL_NAME = "dino_vitb8"

CITYSCAPE_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

SKY_CLASS_ID = CITYSCAPE_CLASSES.index("sky")

NUM_SELECT_PCA = 2000000
PCA_DIM = 64
SEED = 1234
BACKEND = "sklearn"
PRECISION = np.float16

def get_pca_color(features: Tensor, m: float = 3.0, backend="sklearn"):
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors
        m : a hyperparam controlling how many std dev outside for outliers

    Returns:
        Tensor: Colored image
    """

    if backend == "torch":
        torch.manual_seed(SEED)
        _, _, reduction_matrix = torch.pca_lowrank(features, q=3, niter=10)
        _mean = features.mean(dim=0)
    elif backend == "sklearn":
        pca = PCA()
        np.random.seed(SEED)
        pca.fit(features.numpy())
        reduction_matrix = torch.tensor(pca.components_.T[:, :3])
        _mean = torch.tensor(pca.mean_)
        print(f"preserved variance in rgb = {sum(pca.explained_variance_ratio_[:3])}")
    else:
        raise ValueError(f"unknown backend: {backend}")

    features = torch.matmul(features - _mean, reduction_matrix)
    d = torch.abs(features - torch.median(features, dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev

    rgb_ins = features[s[:, 0] < m, :]
    rgb_min, _ = rgb_ins.min(0)
    rgb_max, _ = rgb_ins.max(0)

    return reduction_matrix, rgb_min, rgb_max, _mean

def extract_dino_features(
    rank: int = 0,
    image_filenames: List[str] = [],
    image_shape: Tuple[int, int] = (576, 1024),
    stride: int = 8,
    model_type: str = "dino_vitb8",
    world_size: int = 1,
    verbose: bool = True,
):
    """
    Extracts DINO features from a list of images and saves them to disk.

    Args:
        image_list (List[str]): List of image file paths.
        img_shape (Tuple[int, int], optional): Image shape to resize to. Defaults to (640, 960).
        stride (int, optional): Stride for the ViT extractor. Defaults to 8.
        model_type (str, optional): Type of DINO model to use. Defaults to "dino_vitb8".
        return_pca (bool, optional): Whether to return PCA maps. Defaults to False.
        num_cams (int, optional): Number of cameras. Defaults to 3.

    Returns:
        np.ndarray: Concatenated PCA maps of first num_cams images if return_pca is True, else None.
    """
    device = f"cuda:{rank}"
    image_shape = list(image_shape)
    extractor = ViTExtractor(
        model_type=model_type,
        stride=stride,
        device=device
    )
    
    prep = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_shape),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dino_feature_list = []
    # for ann in tqdm(annotation[rank::world_size], "Extracting Dino features"):
    if verbose and rank == 0:
        pbar = tqdm(desc="Extracting Dino features", total=len(image_filenames))
    chunksize = len(image_filenames) // world_size + 1
    for image_filename in image_filenames[chunksize*rank: min(chunksize*(rank+1), len(image_filenames))]:
        # image_filename = ann['filename']
        images = [prep(Image.open(image_filename).convert("RGB"))]
        preproc_image_lst = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            descriptors = extractor.extract_descriptors(
                preproc_image_lst,
                [11],  # 11 for vit-b and vit-s, 23 for vit-l
                "key",
                include_cls=False,
            )
        descriptors = descriptors.reshape(
            descriptors.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1
        ).squeeze().cpu()
        dino_feature_list.append(descriptors)

        if verbose and rank == 0:
            pbar.update(world_size)

    del extractor
    torch.cuda.empty_cache()

    return dino_feature_list

def pca_reduction(dino_features, sky_masks=None, masks=None, backend="torch"):
    '''
    Do PCA reduction on the dino_features.

    Args:
        dino_features: (N, h, w, c) tensor of dino features
        sky_masks: (N, H, W) tensor of sky masks
        masks: (N, H, W) tensor of masks
    
    Returns:
        pca_dino_features: (N, h, w, PCA_DIM) tensor of PCA-reduced dino features
        pca_results: a dictionary containing PCA parameters
    '''
    print("Doing PCA reduction...")
    start = time()
    N, h, w, c = dino_features.shape
    dino_features = dino_features.flatten(0, -2) # (N*h*w, c)
    if sky_masks is not None:
        sky_masks = F.interpolate(
            sky_masks.reshape(N, 1, H, W).to(torch.uint8), 
            size=(h, w), 
            mode="nearest-exact").to(torch.bool).reshape(N, h, w, 1) # (N, h, w, 1)
    else:
        sky_masks = torch.ones(N, h, w, 1, dtype=torch.bool)
    
    if masks is not None:
        masks = F.interpolate(
            masks.reshape(N, 1, H, W).to(torch.uint8), 
            size=(h, w), 
            mode="nearest-exact").to(torch.bool).reshape(N, h, w, 1) # (N, h, w, 1)
    else:
        masks = torch.ones(N, h, w, 1, dtype=torch.bool)
    
    valid_mask = sky_masks.flatten() & masks.flatten() # (N*h*w,)
    valid_dino_features = dino_features[valid_mask]

    np.random.seed(SEED)
    choosen_idx = np.random.choice(np.arange(len(valid_dino_features)), 
                    size=min(len(valid_dino_features), NUM_SELECT_PCA), replace=False)
    
    if backend == "torch":
        torch.manual_seed(SEED)
        U, S, reduction_matrix = torch.pca_lowrank(valid_dino_features[choosen_idx], q=PCA_DIM, niter=10)
        mean_ = valid_dino_features[choosen_idx].mean(dim=0)
    
    elif backend == "sklearn":
        pca = PCA()
        np.random.seed(SEED)
        pca.fit(valid_dino_features[choosen_idx].numpy())
        reduction_matrix = torch.tensor(pca.components_.T[:, :PCA_DIM])
        mean_ = torch.tensor(pca.mean_)
        print(f"preserved variance = {sum(pca.explained_variance_ratio_[:PCA_DIM])}")
    
    else:
        raise ValueError(f"unknown backend: {backend}")

    pca_dino_features = ((dino_features - mean_) @ reduction_matrix) # (N*h*w, PCA_DIM)
    # normalize to (0, 1)
    _min, _max = pca_dino_features.min(0)[0], pca_dino_features.max(0)[0]
    pca_dino_features = (pca_dino_features - _min) / (_max - _min) # (N*h*w, PCA_DIM)
    pca_dino_features = pca_dino_features.reshape(N, h, w, PCA_DIM)
    
    pca_results = {
        "reduction_matrix": reduction_matrix.numpy(),
        "min": _min.numpy(),
        "max": _max.numpy(),
        "mean": mean_.numpy(),
    }
    print(f"done in {time() - start:.2f}s")
    return pca_dino_features, pca_results

def get_rgb_pca(pca_dino_features, sky_masks=None, masks=None, backend="torch", visualize=False):
    '''
    Get the features-to-rgb maps from the pca_dino_features.

    Args:
        pca_dino_features: (N, h, w, PCA_DIM) tensor of PCA-reduced dino features
        sky_masks: (N, H, W) tensor of sky masks
        masks: (N, H, W) tensor of masks
    
    Returns:
        dino_to_rgb: a dictionary containing PCA parameters for features-to-rgb mapping.
    '''
    N, h, w, c = pca_dino_features.shape
    if sky_masks is not None:
        sky_masks = F.interpolate(
            sky_masks.reshape(N, 1, H, W).to(torch.uint8), 
            size=(h, w), 
            mode="nearest-exact").to(torch.bool).reshape(N, h, w, 1) # (N, h, w, 1)
    else:
        sky_masks = torch.ones(N, h, w, 1, dtype=torch.bool) # True for non-sky pixels
    
    if masks is not None:
        masks = F.interpolate(
            masks.reshape(N, 1, H, W).to(torch.uint8), 
            size=(h, w), 
            mode="nearest-exact").to(torch.bool).reshape(N, h, w, 1) # (N, h, w, 1)
    else:
        masks = torch.ones(N, h, w, 1, dtype=torch.bool)

    # do rgb pca
    valid_mask = sky_masks.flatten() & masks.flatten() # (N*h*w,)
    pca_dino_features = pca_dino_features.flatten(0, -2) # (N*h*w, 64)
    valid_dino_features = pca_dino_features[valid_mask]

    np.random.seed(1234)
    choosen_idx = np.random.choice(np.arange(len(valid_dino_features)), 
                    size=min(len(valid_dino_features), NUM_SELECT_PCA), replace=False)
    
    reduction_matrix_rgb, rgb_min, rgb_max, _mean = get_pca_color(valid_dino_features[choosen_idx], 
                                                                  backend=backend)
    dino_to_rgb = {
        "reduction_matrix": reduction_matrix_rgb.numpy(),
        "rgb_min": rgb_min.numpy(),
        "rgb_max": rgb_max.numpy(),
        "mean": _mean.numpy(),
    }
    pca_dino_features = pca_dino_features.reshape(N, h, w, c)
    if visualize:
        i = 0
        images = (pca_dino_features[i] - dino_to_rgb['mean']).numpy() @ dino_to_rgb['reduction_matrix']
        images = images - dino_to_rgb['rgb_min']
        images = images / (dino_to_rgb['rgb_max'] - dino_to_rgb['rgb_min'])
        images = np.clip(images, 0, 1)
        images_long = (images * 255).astype(np.uint8)
        image = Image.fromarray(images_long, mode='RGB')
        fpath = f"pca_{i}.png"
        image.save(fpath)
    
    return dino_to_rgb

class ViTExtractor:
    # Modified from https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dino_vits8",
        stride: int = 4,
        model: nn.Module = None,
        device: str = "cuda",
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        # print(self.model)
        p = (
            self.model.patch_embed.patch_size
            if isinstance(self.model.patch_embed.patch_size, int)
            else self.model.patch_embed.patch_size[0]
        )
        self.p = p
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        # TODO: make it work better
        if "dinov2" in model_type:
            model = torch.hub.load("facebookresearch/dinov2:main", model_type)
        elif "dino" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        elif "clip" in model_type:
            # TODO: improve this
            model = timm.create_model("vit_base_patch16_clip_224", pretrained=True)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            raise NotImplementedError(
                "Only dino and timm models are supported at the moment."
            )
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess(
        self, image_path, load_size: Union[int, Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        image = Image.open(image_path).convert("RGB")
        # pil_image = image.convert('RGB')
        # if load_size is not None:
        #     pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(load_size),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        prep_img = prep(image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, layers: List[int] = 11, facet: str = "key"
    ) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(
            B, bin_x.shape[1], self.num_patches[0], self.num_patches[1]
        )
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(
                win_size, stride=1, padding=win_size // 2, count_include_pad=False
            )
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros(
            (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
        ).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(
                            x - kernel_size, x + kernel_size + 1, kernel_size
                        ):
                            if i == y and j == x and k != 0:
                                continue
                            if (
                                0 <= i < self.num_patches[0]
                                and 0 <= j < self.num_patches[1]
                            ):
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        bin_x = (
            bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        )
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: List[int],
        facet: str = "key",
        bin: bool = False,
        include_cls: bool = False,
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, layer, facet)
        x = torch.concat(self._feats)
        # if facet == 'token':
        #     x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert (
                not bin
            ), "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = (
                x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
            )  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert (
            self.model_type == "dino_vits8"
        ), f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (
            temp_maxs - temp_mins
        )  # normalize to range [0,1]
        return cls_attn_maps

def dump_dino_features(dino_features, save_filenames):
    assert len(dino_features) == len(save_filenames)
    for i, fpath in enumerate(tqdm(save_filenames, desc="dumping dino files")):
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
        np.savez_compressed(fpath, dino_features[i].astype(PRECISION))
    
    return True

if __name__ == '__main__':
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--mode', type=str, choices=[
        "get_dino", "get_reduction_matrix",
    ], default="get_reduction_matrix")
    parser.add_argument('--skip-exists', action="store_true")
    args = parser.parse_args()
    data_root = args.data_root

    if args.mode == "get_reduction_matrix":
        nusc = NuScenes(version="v1.0-trainval", 
                        dataroot=data_root, 
                        verbose=False)
        
        image_filenames = []
        mask_filenames = []
        sky_filenames = []
        for scene in nusc.scene:
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            scene_name = scene["name"]
            for camera in CAMERAS[:3]:
                sample_data_token = sample['data'][camera]
                sample_data = nusc.get('sample_data', sample_data_token)

                image_filenames.append(os.path.join(data_root, sample_data['filename']))
                sky_filenames.append(os.path.join(
                    data_root, "segmentation", scene_name, camera,
                    sample_data['filename'].split('/')[-1].replace('jpg', 'npz')
                ))
        
        fn = partial(extract_dino_features, image_filenames=image_filenames, image_shape=INPUT_IMAGE_SHAPE, 
                    stride=8, model_type=MODEL_NAME, world_size=args.world_size)
        ctx = mp.get_context("spawn")
        pool = mp.pool.Pool(args.world_size, context=ctx)
        results = pool.map(fn, list(range(args.world_size)))
        pool.close()

        dino_feature_list = []
        for d in results:
            dino_feature_list.extend(d)

        dino_features = torch.stack(dino_feature_list, dim=0) # (N, h, w, c)
        pca_dino_features, pca_results = pca_reduction(dino_features, 
            sky_masks=None, masks=None, backend=BACKEND) # (N, h, w, 64)

        os.makedirs(os.path.join(data_root, "dino_features"), exist_ok=True)
        with open(os.path.join(data_root, "dino_features", "pca_results.pkl"), 'wb') as f:
            pickle.dump(pca_results, f)
        
        skys = []
        
        for filename in sky_filenames:
            seg = np.load(filename, mmap_mode='r')
            if isinstance(seg, np.lib.npyio.NpzFile):
                seg = seg['arr_0']
        
            seg = torch.tensor(seg, dtype=torch.uint8)
            skys.append(~(seg == SKY_CLASS_ID)) # False for sky
        
        skys = torch.stack(skys, dim=0)
        dino_to_rgb = get_rgb_pca(pca_dino_features, 
            sky_masks=skys, masks=None, backend=BACKEND, visualize=False)
        with open(os.path.join(data_root, "dino_features", "dino_to_rgb.pkl"), 'wb') as f:
            pickle.dump(dino_to_rgb, f)
    
    elif args.mode == "get_dino":
        nusc = NuScenes(version="v1.0-trainval", 
                        dataroot=data_root, 
                        verbose=False)
        with open(os.path.join(data_root, "dino_features", "pca_results.pkl"), 'rb') as f:
            pca_results = pickle.load(f)
        reduction_matrix = torch.tensor(pca_results["reduction_matrix"])
        _min = torch.tensor(pca_results["min"])
        _max = torch.tensor(pca_results["max"])
        _mean = torch.tensor(pca_results["mean"])

        dump_pool = ThreadPoolExecutor(2)
        future = None
        
        scenes = [s["name"] for s in nusc.scene]
        scenes = sorted(scenes)
        for scene_name in tqdm(scenes, desc="Extracting dino features for all scenes"):
            target_path = os.path.join(data_root, f"dino_features/{scene_name}/")
            with open(os.path.join(data_root, f"PreSight/{scene_name}.pkl"), 'rb') as f:
                annotations = pickle.load(f)

            image_filenames = [a["filename"] for a in annotations]
            save_filenames = [a["dino_filename"] for a in annotations]
            # check exists
            if (args.skip_exists and os.path.exists(target_path) and 
                len(os.listdir(target_path)) == len(save_filenames)):
                sleep(0.02)
                continue

            ctx = mp.get_context("spawn")
            pool = mp.pool.Pool(args.world_size, context=ctx)
            fn = partial(extract_dino_features, image_filenames=image_filenames, image_shape=INPUT_IMAGE_SHAPE, 
                        stride=STRIDE, model_type=MODEL_NAME, world_size=args.world_size, verbose=True)
            
            results = pool.map(fn, list(range(args.world_size)))
            pool.close()
            dino_feature_list = []
            for d in results:
                dino_feature_list.extend(d)
            dino_features = torch.stack(dino_feature_list)
            dino_features = (dino_features - _mean) @ reduction_matrix
            dino_features = (dino_features - _min) / (_max - _min)
            dino_features = dino_features.numpy().astype(np.float16)
            
            if future is not None:
                future.result()
            future = dump_pool.submit(dump_dino_features, dino_features, save_filenames)
        
        dump_pool.shutdown()
    else:
        raise ValueError(f"unknown mode {args.mode}")
