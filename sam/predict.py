import os

import numpy as np

import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from skimage.draw import polygon as draw_polygon

from datasets import norm_depth_unstd
from sam.model import SegmentAnythingModel

def polygon_to_mask(
        image_shape: Tuple[int, int], 
        polygon_points: List[List[float]]
        ) -> np.ndarray:
    """
    Converts a list of polygon vertices to a binary mask.
    """

    if not polygon_points:
        return np.zeros(image_shape, dtype=np.uint8)
    
    points_array = np.array(polygon_points)
    # scikit-image's draw.polygon expects (row, col) coordinates
    rr, cc = draw_polygon(
        points_array[:, 1], 
        points_array[:, 0], 
        shape=image_shape
        )
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask

def calculate_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) for two binary masks."""
    if gt_mask.shape != pred_mask.shape:
        gt_mask = gt_mask.squeeze(-1)
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    # print(np.sum(intersection), np.sum(union))
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou_score

def show_mask(mask: np.ndarray, ax, color: List[float]):
    """
    Helper function to display a mask on a plot with transparency.
    This version is more robust and avoids the reshape error.
    """
    # Squeeze the mask to remove any singleton dimensions (e.g., from (H, W, 1) to (H, W))
    mask = np.squeeze(mask)
    
    # Ensure the mask is a boolean array for indexing
    mask = mask.astype(bool)
    
    # Create a blank RGBA image
    h, w = mask.shape
    mask_image = np.zeros((h, w, 4), dtype=np.float32)
    
    # Where the mask is True, apply the specified color
    mask_image[mask] = color
    
    # Display the result
    ax.imshow(mask_image)

class SAMPredictor:
    def __init__(
            self, 
            sam_model_type: str, 
            sam_checkpoint: str, 
            debug: bool = False
            ):
        self.debug = debug

        try:
            print(f"Initializing Segment Anything Model (type: {sam_model_type})...")
            if sam_model_type not in sam_checkpoint:
                raise ValueError(f"Checkpoint for {sam_model_type} not found in config.")
            checkpoint_path = sam_checkpoint[sam_model_type]
            self.sam_model = SegmentAnythingModel(
                model_type=sam_model_type,
                checkpoint_path=checkpoint_path
            )
        except Exception as e:
            print(f"Fatal Error: Could not initialize SAM model: {e}")
            self.sam_model = None
        
    def _debug_plot(
            self, 
            image, 
            gt_mask, pred_mask,
            pos_prompt, neg_prompt,
            title):
        """Shows an immediate plot for debugging a single prediction."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(title, fontsize=16)
        axs[0].imshow(image, cmap='gray')

        axs[0].set_title("Ground Truth Mask")
        axs[0].axis('off')
        
        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(
            pred_mask, 
            cmap='jet', 
            alpha=0.8
            )
        axs[1].scatter(
            pos_prompt[:, 0],
            pos_prompt[:, 1],
            color='green',
            marker='*',
            s=200,
            edgecolor='white'
            )
        if pos_prompt is not None and pos_prompt.ndim > 0:
            axs[1].scatter(
                pos_prompt[:, 0],
                pos_prompt[:, 1],
                color='green',
                marker='o',
                s=200,
                edgecolor='white'
            )
        if neg_prompt is not None and neg_prompt.ndim > 0:
            axs[1].scatter(
                neg_prompt[:, 0], 
                neg_prompt[:, 1], 
                color='red', 
                marker='x', 
                s=200, 
                edgecolor='white'
            )
        axs[1].set_title("Predicted Mask")
        axs[1].axis('off')
        
        plt.show() # This will pause execution until the plot is closed

    def run_evaluation(
            self, 
            dataset, 
            patch_indices: Optional[List[int]] = None,
            use_negative_prompts: bool = False
            ):
        """
        Runs the full SAM evaluation pipeline on a dataset.
        Can process the entire dataset or a specific list of patch indices.
        Input:  
            dataset: The dataset to evaluate.
            patch_indices: Specific patch indices to process.
        """
        if not self.sam_model: 
            return
        print("\n===== Running Zero-Shot Segmentation Evaluation with SAM =====")

        indices_to_process = patch_indices if patch_indices \
            is not None else range(len(dataset))

        for idx in indices_to_process:
            data_pair = dataset.get_patch_by_index(idx) \
                if patch_indices else dataset[idx]
            if not data_pair: 
                continue

            inpainted_img = data_pair['inpainted_depth']
            uninpainted_img = data_pair['original_depth']
            mask_inpainting = data_pair['mask_inpainting']
            mask_boundary = data_pair['mask_boundary']

            # Normalize images to [0, 1] for Canny and visualization
            uninpainted_norm = norm_depth_unstd(
                uninpainted_img, 
                mask_inpainting=mask_inpainting, 
                mask_boundary=mask_boundary,
                abs_normalize=True
                )
            inpainted_norm = inpainted_img.copy()  # Already normalized to [0, 1]

            uninpainted_norm = np.mean(
                uninpainted_norm, axis=2
                )  # Convert to grayscale
            # print('uninpainted_norm shape:', uninpainted_norm.shape)
            inpainted_norm = np.mean(
                inpainted_img, axis=2
                )  # Convert to grayscale
            
            # Define saved loc
            data_path = os.path.dirname(data_pair['path'])
            instance_path = os.path.dirname(data_path)
            instance_name = os.path.basename(instance_path)
            sigma_y = dataset.sigma_y
            if sigma_y == 0.0:
                sam_output_folder_name = f"{instance_name}_sam_output"
            else:
                sam_output_folder_name = f"{instance_name}_sam_output_{sigma_y}"

            if use_negative_prompts:
                sam_output_folder_name += "_neg"
            
            save_dir = os.path.join(
                os.path.dirname(instance_path), 
                sam_output_folder_name
                )
            os.makedirs(save_dir, exist_ok=True)
            # patch name is 'patch_{idx}.npz', want to remove .npz
            patch_base_name =  os.path.splitext(
                data_pair['patch_base_name']
                )[0]
            save_path = os.path.join(save_dir, patch_base_name)
            results_for_npz = {}

            for label_idx, polygon_points in enumerate(
                data_pair['labels']
                ):
                if not polygon_points:
                    raise ValueError("Polygon points are empty")

                gt_mask = polygon_to_mask(
                    uninpainted_img.shape, 
                    polygon_points
                    )
                points_array = np.array(
                    polygon_points
                    )
                
                # Generate Prompts (simple version)
                positive_prompt = points_array.mean(
                    axis=0
                    ).astype(int).reshape(1, 2)
                # Example negative prompt: a point outside the bounding box
                negative_prompt = None
                if use_negative_prompts:
                    # Generate a random negative prompt outside the GT mask
                    while True:
                        y, x = np.random.randint(0, gt_mask.shape[0]),\
                               np.random.randint(0, gt_mask.shape[1])
                        if gt_mask[y, x] == 0:
                            negative_prompt = np.array([[x, y]])
                            break

                # Run predictions
                pred_mask_uninpainted = self.sam_model.predict(
                    uninpainted_img, 
                    positive_prompt, negative_prompt
                    )
                iou_uninpainted = calculate_iou(
                    gt_mask, pred_mask_uninpainted
                    )

                pred_mask_inpainted = self.sam_model.predict(
                    inpainted_img, 
                    positive_prompt, negative_prompt
                )

                iou_inpainted = calculate_iou(
                    gt_mask, 
                    pred_mask_inpainted,

                    )
                if self.debug:
                    self._debug_plot(
                        inpainted_norm, 
                        gt_mask, 
                        pred_mask_inpainted,
                        positive_prompt,
                        negative_prompt, 
                        f"Debug: Patch {idx}, Label {label_idx}"
                        )
                    
                # Store results for this label, prefixing keys with label index
                results_for_npz[f'gt_mask_{label_idx}'] = gt_mask
                results_for_npz[f'pred_mask_uninpainted_{label_idx}'] = pred_mask_uninpainted
                results_for_npz[f'pred_mask_inpainted_{label_idx}'] = pred_mask_inpainted
                results_for_npz[f'iou_uninpainted_{label_idx}'] = iou_uninpainted
                results_for_npz[f'iou_inpainted_{label_idx}'] = iou_inpainted
                results_for_npz[f'positive_prompt_{label_idx}'] = positive_prompt
                results_for_npz[f'negative_prompt_{label_idx}'] = negative_prompt
            if results_for_npz:
                np.savez(save_path, **results_for_npz)
                print(f"  Saved SAM results to {save_path}")