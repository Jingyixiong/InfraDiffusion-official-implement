import os
import numpy as np
import glob

from torch.utils.data import Dataset
import torch.utils.data as data
# from datasets import norm_depth_std, norm_depth_unstd 

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

class IRDataset(Dataset):
    """A general dataset to load rendered depth maps for inpainting."""
    def __init__(
            self, 
            root: str,
            patch_size: tuple, 
            infra_name: str = None,
            norm_method: str = 'unstd',
            image_size: tuple = (256, 256), 
            abs_normalize: bool = True,
            require_labels: bool = False 
            ):
        self.root_dir = root
        self.patch_size = patch_size
        self.infra_name = infra_name
        self.norm_method = norm_method
        self.image_size = image_size  # Not used in this dataset, but can be useful for future extensions
        self.abs_normalize = abs_normalize
        self.require_labels = require_labels
        self.files = []
        self._scan_directory()

    def _scan_directory(self):
        """Finds all rendered patch files and their corresponding camera params."""
        size_str = "_".join(map(str, self.patch_size))
        rendered_folder_name = f"rendered_{size_str}"

        infra_pattern = self.infra_name if self.infra_name\
            is not None else '*'
        pattern = os.path.join(
            self.root_dir, 
            infra_pattern, 
            '*', '*', 
            rendered_folder_name
            )
        for rendered_dir in sorted(glob.glob(pattern)):
            for f in sorted(os.listdir(rendered_dir)):
                if f.startswith('patch_') and \
                    f.endswith('.npz') and \
                        'cam_params' not in f:
                    patch_path = os.path.join(
                        rendered_dir, f
                        )
                    cam_path = os.path.join(
                        rendered_dir, 
                        f.replace('.npz', '_cam_params.npz')
                        )
                    
                    if not os.path.exists(cam_path):
                        continue
                    # extract data only if GT exists
                    if self.require_labels:
                        instance_path = os.path.dirname(
                            rendered_dir
                            )
                        instance_name = os.path.basename(
                            instance_path
                            )
                        label_folder_path = os.path.join(
                            os.path.dirname(instance_path), 
                            f"{instance_name}_label"
                            )
                        label_filename = os.path.basename(
                            patch_path
                            ).replace('.npz', '.json')
                        label_path = os.path.join(
                            label_folder_path, 
                            label_filename
                            )
                        
                        # If labels are required, skip this patch if the label file doesn't exist
                        if not os.path.exists(label_path):
                            continue

                    self.files.append(
                            {'patch': patch_path, 
                             'cam': cam_path}
                             )
                    # print(f"Found patch: {patch_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        paths = self.files[idx]
        img_data = np.load(paths['patch'])
        cam_param = np.load(paths['cam'])
        
        depth_map = img_data["depth_map"]

        mask_inpainting = img_data["mask_inpainting"]
        mask_boundary = img_data["mask_boundary"]
        # make sure masks has consistent data type
        mask_inpainting = img_data["mask_inpainting"].astype(np.uint8)
        mask_boundary = img_data["mask_boundary"].astype(np.uint8)
        
        if self.norm_method == 'std':
            norm_depth_map = norm_depth_std(
                depth_map, 
                mask_boundary=mask_boundary,
                mask_inpainting=mask_inpainting,
                abs_normalize=self.abs_normalize 
                )
        elif self.norm_method == 'unstd':
            norm_depth_map = norm_depth_unstd(
                depth_map, 
                mask_boundary=mask_boundary,
                mask_inpainting=mask_inpainting,
                abs_normalize=self.abs_normalize 
                )
        else:
            raise ValueError(f"Unknown norm_method: {self.norm_method}")

        if norm_depth_map.ndim == 2:
            norm_depth_map = np.expand_dims(
                norm_depth_map, axis=2
                )
        norm_depth_map = np.repeat(
            norm_depth_map, 3, axis=2
            )

        # Return the patch path to be used for saving the output
        return norm_depth_map, depth_map, \
            mask_inpainting, mask_boundary, \
                cam_param, paths['patch']


if __name__ == "__main__":
    # root_dir = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/tunnels/wheatly_tunnel'  # Update with your actual path
    # root_dir = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/masonry_bridges'
    root_dir = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/tunnels/wheatly_tunnel'
    # Example usage
    dataset = IRDataset(
        root=root_dir,
        patch_size=(0.8, 0.8, 0.5),
        infra_name='S-20',
        norm_method='unstd',
        abs_normalize=True,
        require_labels=True
    )
    print(len(dataset))
    norm_depth_map, depth_map, \
            mask_inpainting, mask_boundary, \
                cam_params, path = dataset[32]
    # print(f"norm_depth_map shape: {norm_depth_map.shape}")
    # print(f"depth_map shape: {depth_map.shape}")
    # print(f"mask_inpainting shape: {mask_inpainting.shape}")
    # print(f"mask_boundary shape: {mask_boundary.shape}")
    # print(f"Camera parameters: {cam_params}")
    # print(f"Patch path: {path}")
    # print(len(dataset))
    # # visualize the norm_depth_map
    # import matplotlib.pyplot as plt
    # print('range of values in norm_depth_map:', 
    #       np.min(norm_depth_map), np.max(norm_depth_map))
    # plt.imshow(norm_depth_map[..., 0], cmap='gray')
    # plt.title('Normalized Depth Map')
    # plt.axis('off')
    # plt.show()

    # for i in range(len(dataset)):
    #     norm_depth_map, depth_map, \
    #         mask_inpainting, mask_boundary, \
    #             cam_params, path = dataset[i]
    #     print(f"Sample {i}: norm_depth_map shape: {norm_depth_map.shape}")