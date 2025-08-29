import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import numpy as np
import argparse

from PIL import Image
from datasets.eval import EvaluationDataset, \
    SAMResultsDataset

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_postprocessing_dataset(config, sigma_y=0.16):
    """
    Creates a dataset for comparing original and inpainted depth maps.
    
    Args:
        root_dir (str): Root directory containing the data.
        patch_size (tuple): Size of the patches.
        infra_name (Optional[str]): Specific infrastructure name to filter data.
        
    Returns:
        ComparisonTaskDataset: The dataset instance.
    """
    # config = dict2namespace(config)
    active_infra_name = config.data.active_infra
    try:
        infra_config = config.infra.masonry_bridges\
        if active_infra_name == 'masonry_bridges' else \
            config.infra.tunnel
        root_dir = infra_config.root
        patch_size = infra_config.patch_size
        infra_name = infra_config.infra_name
    except KeyError:
            raise KeyError(
                f"Configuration for '{active_infra_name}'"+
                "not found in the 'infra' section of your .yml file."
                )  
    return EvaluationDataset(
        root=root_dir,
        patch_size=patch_size,
        infra_name=infra_name,
        sigma_y=sigma_y
    )

# def get_samresults_dataset(
#         config, 
#         use_negative_prompts, 
#         sigma_y=0.16
#         ):
#     # Implement the function to load the SAM results dataset
#     active_infra_name = config.data.active_infra
#     try:
#         infra_config = config.infra.masonry_bridges\
#         if active_infra_name == 'masonry_bridges' else \
#             config.infra.tunnel
#         root_dir = infra_config.root
#         patch_size = infra_config.patch_size
#         infra_name = infra_config.infra_name
#     except KeyError:
#             raise KeyError(
#                 f"Configuration for '{active_infra_name}'"+
#                 "not found in the 'infra' section of your .yml file."
#                 )  
#     return SAMResultsDataset(
#         root_dir=root_dir,
#         patch_size=patch_size,
#         infra_name=infra_name,
#         use_negative_prompts=use_negative_prompts,
#         sigma_y=sigma_y
#     )

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "InfraDataset":
        from datasets.infra import IRDataset
        # args.path_y in main.py represents the path to store all datasets

        norm_method = config.data.norm_method    # Normalize depth map to [-1, 1]
        active_infra_name = config.data.active_infra
        print(f"Loading active infrastructure dataset: {active_infra_name}")
        try:
            infra_config = config.infra.masonry_bridges\
            if active_infra_name == 'masonry_bridges' else \
                config.infra.tunnel
            root_dir = infra_config.root
            patch_size = infra_config.patch_size
            infra_name = infra_config.infra_name
        except KeyError:
            raise KeyError(
                f"Configuration for '{active_infra_name}'"+
                "not found in the 'infra' section of your .yml file."
                )
        dataset = IRDataset(
            root=root_dir,
            patch_size=patch_size,
            image_size=config.data.image_size,
            require_labels=config.data.require_labels,
            infra_name=infra_name,
            norm_method=norm_method
        )
        test_dataset = dataset  # same set for now, or use different split logic

    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.general.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.general.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.general.rescaled:
        X = 2 * X - 1.0
    elif config.general.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.general.logit_transform:
        X = torch.sigmoid(X)
    elif config.general.rescaled:
        X = (X + 1.0) / 2.0

    return X
    # return torch.clamp(X, 0.0, 1.0)


def norm_depth_std(depth, mask_boundary, mask_inpainting, abs_normalize=False):
    """
    Normalize depth to roughly [-1, 1] using the 2nd and 98th percentiles.
    The range of the input obey the diffusion model's requirements.
    Masks are used to ignore invalid pixels.
    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        mask_boundary (np.ndarray): Boundary mask of shape (H, W) with values 0 or 1.
        mask_inpainting (np.ndarray): Inpainting mask of shape (H, W) with values 0 or 1.
        abs_normalize (bool): If True, normalize with max and min values. Else, use percentiles.
    Outputs:
        np.ndarray: Normalized depth map of shape (H, W, 1)
    """
    dtype = depth.dtype
    valid_mask = (mask_inpainting == 1)
    # print('mask_inpainting shape:', mask_inpainting)
    # print('valid pixels:', np.sum(valid_mask))
    depth_norm = np.zeros_like(depth, dtype=dtype)
    # print('Depth norm shape:', depth_norm.shape)
    # print(valid_mask)
    if abs_normalize:
        d_max, d_min = np.max(depth[valid_mask]), np.min(depth[valid_mask])
        # Normalize to [-1, 1]
        scale = d_max - d_min if d_max != d_min else 1.0
        # Scale to [0, 1] first, then multiply by 2 and subtract 1
        depth_norm[valid_mask] = 2.0 * ((depth[valid_mask] - d_min) / scale) - 1.0
    else:
        d2 = np.percentile(depth[valid_mask], 2)
        d98 = np.percentile(depth[valid_mask], 98)
        # Normalize to [-1, 1]
        scale = d98 - d2 if d98 != d2 else 1.0
        depth_norm[valid_mask] = 2.0 * ((depth[valid_mask] - d2) / scale) - 1.0 
        np.clip(depth_norm, -1.0, 1.0, out=depth_norm)

    return depth_norm.astype(dtype)

def inv_norm_depth_std(
        depth_norm, original_depth, 
        mask_boundary, mask_inpainting, 
        abs_normalize=False
        ):
    """
    Invert depth normalization based on 'norm_depth_std'.

    Args:
        depth_norm (np.ndarray): Normalized depth map, in [-1, 1].
        original_depth (np.ndarray): Original depth map to calculate stats.
        mask_boundary (np.ndarray): Boundary mask (H, W).
        mask_inpainting (np.ndarray): Inpainting mask (H, W).
        abs_normalize (bool): If True, uses min/max instead of percentiles.

    Returns:
        np.ndarray: Reconstructed depth map with original scale.
    """
    dtype = depth_norm.dtype
    # The valid mask should use the original boundary, not the inpainting mask,
    # to ensure the scaling factors (min/max) are consistent with the norm_depth function.
    valid_mask = (mask_boundary == 1)
    depth_rec = np.zeros_like(depth_norm, dtype=dtype)
    
    # First, scale the [-1, 1] data back to the [0, 1] range
    depth_0_to_1 = (depth_norm + 1.0) / 2.0

    if abs_normalize:
        d_max = np.max(original_depth[valid_mask])
        d_min = np.min(original_depth[valid_mask])
        scale = d_max - d_min if d_max != d_min else 1.0
        depth_rec[valid_mask] = depth_0_to_1[valid_mask] * scale + d_min
    else:
        d2 = np.percentile(original_depth[valid_mask], 2)
        d98 = np.percentile(original_depth[valid_mask], 98)
        scale = d98 - d2 if d98 != d2 else 1.0
        depth_rec[valid_mask] = depth_0_to_1[valid_mask] * scale + d2

    return depth_rec.astype(dtype)

def norm_depth_unstd(depth, mask_boundary, mask_inpainting, abs_normalize=False):
    """
    Normalize depth to roughly [0, 1] using the 2nd and 98th percentiles.
    Masks are used to ignore invalid pixels.
    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        mask_boundary (np.ndarray): Boundary mask of shape (H, W) with values 0 or 1.
        mask_inpainting (np.ndarray): Inpainting mask of shape (H, W) with values 0 or 1.
        abs_normalize (bool): If True, normalize with max and min values. Else, use percentiles.
    """
    dtype = depth.dtype
    valid_mask = (mask_inpainting == 1)
    # print('valid pixels:', np.sum(valid_mask))
    depth_norm = np.zeros_like(depth, dtype=dtype)
    # Safety check: if there are no valid pixels, return a zero array
    # to prevent the "zero-size array" error.
    if not np.any(valid_mask):
        return depth_norm
    
    if abs_normalize:
        d_max, d_min = np.max(depth[valid_mask]), np.min(depth[valid_mask])
        scale = d_max - d_min if d_max != d_min else 1.0
        # Normalize to [0, 1]
        depth_norm[valid_mask] = (depth[valid_mask] - d_min) / scale
    else:
        d2 = np.percentile(depth[valid_mask], 2)
        d98 = np.percentile(depth[valid_mask], 98)
        # Normalize to [0, 1]
        depth_norm[valid_mask] = (depth[valid_mask] - d2) / (d98 - d2)  

    return depth_norm.astype(dtype)

def inv_norm_depth_unstd(
    depth_norm, original_depth, 
    mask_boundary, mask_inpainting,
    abs_normalize=False
    ):

    """
    Invert depth normalization performed by 'norm_depth_unstd'.
    Args:
    depth_norm (np.ndarray): Normalized depth map of shape (H, W, 1), in [0, 1].
    original_depth (np.ndarray): Original depth map, [H, W] or
    [H, W, 1], or [H, W, 3].
    mask_boundary (np.ndarray): Boundary mask (H, W), 0 or 1.
    mask_inpainting (np.ndarray): Inpainting mask (H, W), 0 or 1.
    abs_normalize (bool): If True, uses min/max instead of percentiles.

    Returns:

    np.ndarray: Reconstructed depth map with original scale, shape (H, W, 1).

    """

    dtype = depth_norm.dtype
    valid_mask = (mask_boundary == 1) & (mask_inpainting == 1)
    depth_rec = np.zeros_like(depth_norm, dtype=dtype)

    if abs_normalize:
        d_max = np.max(original_depth[valid_mask])
        d_min = np.min(original_depth[valid_mask])
        depth_rec[valid_mask] = depth_norm[valid_mask] * (d_max - d_min) + d_min
    else:
        d2 = np.percentile(original_depth[valid_mask], 2)
        d98 = np.percentile(original_depth[valid_mask], 98)
        depth_rec[valid_mask] = depth_norm[valid_mask] * (d98 - d2) + d2
    return depth_rec.astype(dtype)

def resize_depth_tensor(depth, target_size):
    ''' Resize depth tensor to target size using bilinear interpolation.
    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        target_size (tuple): Target size (H, W) for resizing.
    '''
    if np.ndim(depth) != 2:
        H, W, _ = depth.shape
    else:
        H, W = depth.shape
    depth_tensor = torch.from_numpy(depth).view(1, 1, H, W).float()  # (1, 1, H, W)
    depth_resized = F.interpolate(depth_tensor, size=target_size, mode='bilinear', align_corners=False)
    return depth_resized.squeeze(0).view(target_size[0], target_size[1], 1)  # (1, H, W)

def resize_binary_mask(mask_np, target_size):
    ''' Resize binary mask to target size using nearest neighbor interpolation(make sure they are 0 or 1).
    Args:
        mask_np (np.ndarray): Binary mask of shape (H, W) with values 0 or 1.
        target_size (tuple): Target size (H, W) for resizing.
    '''
    if np.ndim(mask_np) != 2:
        H, W, _ = mask_np.shape
    else:
        H, W = mask_np.shape
    mask_tensor = torch.from_numpy(mask_np).view(1, 1, H, W).float()  # (1, 1, H, W)
    resized = F.interpolate(mask_tensor, size=target_size, mode='nearest')     # No aliasing for binary
    return resized.view(target_size[0], target_size[1], 1)  # (1, H_out, W_out)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace