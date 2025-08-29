import os
import glob

import numpy as np
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from typing import Optional, Dict, Any
from torch.utils.data import Dataset

class EvaluationDataset(Dataset):
    """
    A unified dataset to load pairs of original rendered data, their
    corresponding inpainted results, and ground-truth labels for evaluation tasks.
    """
    def __init__(
            self, 
            root: str, 
            patch_size: tuple, 
            infra_name: Optional[str] = None,
            sigma_y: float = 0.0,
            use_boundary_constraint: bool = True
            ):
        self.root_dir = root
        self.patch_size = patch_size
        self.infra_name = infra_name
        self.sigma_y = sigma_y
        self.use_boundary_constraint = use_boundary_constraint
        self.file_sets = []
        self._scan_directory()

    def _scan_directory(self):
        """
        Finds all inpainted files and pairs them
        with their original data and labels.
        """
        size_str = "_".join(map(str, self.patch_size))
        if self.sigma_y == 0.0:    # no noise in source img
            inpainting_folder_name = f"inpainting_{size_str}"
        else:
            inpainting_folder_name = f"inpainting_{size_str}_{self.sigma_y}"
        if not self.use_boundary_constraint:
            # Add suffix for the vanilla DDNM
            inpainting_folder_name += "_naive" 

        rendered_folder_name = f"rendered_{size_str}"
        patch_folder_name = f"patch_{size_str}"

        infra_pattern = self.infra_name if \
            self.infra_name is not None else '*'
        pattern = os.path.join(
            self.root_dir, 
            infra_pattern, 
            '*', '*', 
            inpainting_folder_name, 
            'inpainted_*.npz'
            )
        # print(pattern)
        # print(glob.glob(pattern))
        
        for inpainted_path in sorted(glob.glob(pattern)):
            original_filename = os.path.basename(
                inpainted_path
                ).replace('inpainted_', 'patch_')
            # print(inpainted_path)
            # print(original_filename)
            original_path = inpainted_path.replace(
                inpainting_folder_name, 
                rendered_folder_name
                ).replace(os.path.basename(inpainted_path), original_filename)
            # print('original_path:', original_path)
            cam_param_path = original_path.replace(
                '.npz', '_cam_params.npz'
                )
            
            # Look for the corresponding JSON label file
            instance_path = os.path.dirname(
                os.path.dirname(original_path)
                )
            instance_name = os.path.basename(instance_path)
            label_folder_path = os.path.join(
                os.path.dirname(instance_path), 
                f"{instance_name}_label"
                )
            label_filename = original_filename.replace(
                '.npz', '.json'
                )
            label_path = os.path.join(
                label_folder_path, label_filename
                )
            # Only add the data to the dataset if all corresponding files exist
            # print(label_path)
            # print(os.path.exists(label_path))

            if os.path.exists(original_path) and \
               os.path.exists(cam_param_path) and \
               os.path.exists(label_path):
                # print(label_path)
                # print(os.path.exists(label_path))
                self.file_sets.append({
                    'original_rendered': original_path,
                    'inpainted': inpainted_path,
                    'cam_params': cam_param_path,
                    'label': label_path,  # Add the path to the label file
                    'patch_base_name': original_filename
                })
            else:
                continue

            # Uncomment the following lines if you want to include a check for the original file
            # if os.path.exists(original_path)\
            #       and os.path.exists(cam_param_path):
            #     self.file_sets.append({
            #         'original_rendered': original_path,
            #         'inpainted': inpainted_path,
            #         'cam_params': cam_param_path,
            #         'label': label_path,  # Add the path to the label file
            #         'patch_base_name': original_filename
            #     })

    def __len__(self):
        return len(self.file_sets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        paths = self.file_sets[idx]
        
        original_data = np.load(paths['original_rendered'])
        inpainted_data = np.load(paths['inpainted'])
        cam_params = np.load(paths['cam_params'])
        patch_base_name = paths['patch_base_name']
        
        # --- NEW: Load and parse the JSON label file ---
        labels = []
        try:
            with open(paths['label'], 'r') as f:
                label_data = json.load(f)
                # Extract the polygon points from each shape
                labels = [shape['points'] for shape in label_data.get('shapes', [])]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # print(f"Warning: Could not load or parse label file {paths['label']}. Error: {e}")
            # Return empty list if labels can't be loaded
            labels = []

        return {
            'original_depth': original_data['depth_map'],
            'mask_inpainting': original_data['mask_inpainting'],
            'mask_boundary': original_data['mask_boundary'],
            'inpainted_depth': inpainted_data['inpainted_depth'],
            'cam_params': dict(cam_params),
            'labels': labels,       # Add the list of label polygons
            'patch_base_name': patch_base_name,
            'path': paths['inpainted']
        }

    def get_patch_by_index(self, patch_index: int) -> Optional[Dict[str, Any]]:
        """
        Finds and loads a specific patch by its original index number.
        """
        target_filename = f"patch_{patch_index}.npz"
        
        for i, file_set in enumerate(self.file_sets):
            # print(f"Checking {file_set['original_rendered']} against {target_filename}")
            if os.path.basename(file_set['original_rendered']) == target_filename:
                return self.__getitem__(i)
        
        print(f"Warning: Patch with index {patch_index} not found in the dataset.")
        return None
    
    def copy_to(self, destination_root: str):
        """
        Copies all files associated with the dataset to a new directory,
        preserving the hierarchical structure.
        
        Args:
            destination_root (str): The new root directory to copy files to.
        """
        print(f"\n===== Copying {len(self.file_sets)} file sets to {destination_root} =====")
        if not self.file_sets:
            print("No files found to copy.")
            return

        path_keys_to_copy = [
            'original_rendered', 'inpainted', 
            'cam_params', 'label'
            ]

        for file_set in self.file_sets:
            # For each file, determine its relative path from the original root
            for key in path_keys_to_copy:
                if key in file_set:
                    src_path = file_set[key]
                    relative_path = os.path.relpath(src_path, self.root_dir)
                    dest_path = os.path.join(destination_root, relative_path)
                    
                    # Ensure the destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Copy the file
                    shutil.copy(src_path, dest_path)
        
        print("Copying complete.")

        
        print("Copying complete.")

class SAMResultsDataset(Dataset):
    '''
    Extract the predict results from SAM.
    '''
    def __init__(
            self, 
            root: str, 
            patch_size: tuple, 
            infra_name: Optional[str] = None,
            use_negative_prompts: bool = False,
            sigma_y: float=0.16
            ):
        self.root_dir = root
        self.patch_size = patch_size
        self.infra_name = infra_name
        self.use_negative_prompts = use_negative_prompts
        self.sigma_y = sigma_y
        self.result_files = []
        self._scan_directory()

    def _scan_directory(self):
        if self.sigma_y == 0.0:
            sam_output_suffix = "_sam_output"
        else:
            sam_output_suffix = f"_sam_output_{self.sigma_y}"
            
        if self.use_negative_prompts:
            sam_output_suffix += "_neg"

        infra_pattern = self.infra_name if \
            self.infra_name is not None else '*'
        pattern = os.path.join(
            self.root_dir, 
            infra_pattern, '*', 
            f'*{sam_output_suffix}', '*.npz'
            )
        self.result_files = sorted(glob.glob(pattern))

    def __len__(self): 
        return len(self.result_files)

    def __getitem__(self, idx):
        result_path = self.result_files[idx]
        return {
            'sam_results': dict(
                np.load(
                    result_path, 
                    allow_pickle=True
                )), 
            'path': result_path
            }

    def get_patch_by_index(
            self, patch_index: int
            ):
        """
        Finds and loads a specific SAM result by its original patch index.
        """
        target_filename = f"patch_{patch_index}.npz"
        for result_path in self.result_files:
            if os.path.basename(result_path) == target_filename:
                return {
                    'sam_results': dict(np.load(
                    result_path, 
                    allow_pickle=True
                )), 
                    'path': result_path
                    }
        print(f"Warning: SAM result for patch index {patch_index} not found.")
        return None

if __name__ == '__main__':
    test_root = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/tunnels/wheatly_tunnel/S-15/pier/0_label/patch_158.json'
    root_dir = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/masonry_bridges'  # Update with your actual path
    root_dir = '/home/yj401/jing_projects_ubuntu/code_hub/official-2D-3D-rendering/output/tunnels/wheatly_tunnel'  # Example path
    patch_size = (0.8, 0.8, 0.5)  # Example patch size
    # infra_name = 'begc'  # Example infrastructure name

    infra_name = 'S-15'
    noise_level = [0.0, 0.04, 0.08, 0.16]
    # root_dir = 'output_cooperative/masonry_bridges'

    dataset = EvaluationDataset(
            root=root_dir, 
            patch_size=patch_size, 
            infra_name=infra_name,
            sigma_y=0.16,
            use_boundary_constraint=False
            )
    print(len(dataset))
    print(dataset.sigma_y)
    print(dataset[0]['patch_base_name'])