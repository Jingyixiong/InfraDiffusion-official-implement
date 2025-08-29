import os
import numpy as np
import matplotlib.pyplot as plt

from typing import  List


from datasets import norm_depth_unstd

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


def calculate_and_print_miou(dataset):
    """
    Calculates and prints the mean Intersection over Union (mIoU) for each
    infrastructure found in the results dataset.
    """
    print("\n===== STAGE 3: Calculating mIoU Scores =====")
    
    # Dictionary to hold lists of patch-level average IoUs for each infra
    infra_scores_uninpainted = {}
    infra_scores_inpainted = {}

    for i in range(len(dataset)):
        data = dataset[i]
        
        # Extract infrastructure name from the path
        # e.g., '.../output/chelmsford/arch/0_sam_output/patch_0.npz' -> 'chelmsford'
        parts = data['path'].split(os.path.sep)
        infra_name = parts[-4] if len(parts) >= 4 else 'unknown'

        if infra_name not in infra_scores_uninpainted:
            infra_scores_uninpainted[infra_name] = []
            infra_scores_inpainted[infra_name] = []

        num_labels = sum(
            1 for key in data['sam_results'] 
            if key.startswith('gt_mask')
            )
        
        patch_ious_uninpainted = []
        patch_ious_inpainted = []

        # mIoU for all bricks within patch
        for label_idx in range(num_labels):
            patch_ious_uninpainted.append(
                data['sam_results'][f'iou_uninpainted_{label_idx}']
            )
            patch_ious_inpainted.append(
                data['sam_results'][f'iou_inpainted_{label_idx}']
            )

        if patch_ious_uninpainted:
            avg_patch_iou_uninpainted = np.mean(
                patch_ious_uninpainted
                )
            infra_scores_uninpainted[infra_name].append(
                avg_patch_iou_uninpainted
                )
            
        if patch_ious_inpainted:
            avg_patch_iou_inpainted = np.mean(
                patch_ious_inpainted
            )
            infra_scores_inpainted[infra_name].append(
                avg_patch_iou_inpainted
            )

    # Now, calculate and print the final mIoU for each infrastructure
    print("\n--- Mean Intersection over Union (mIoU) Results ---")
    for infra_name in infra_scores_uninpainted.keys():
        miou_uninpainted = np.mean(
            infra_scores_uninpainted[infra_name]
            )
        miou_inpainted = np.mean(
            infra_scores_inpainted[infra_name]
            )
        
        print(f"  Infrastructure: {infra_name}")
        print(f"    mIoU (Uninpainted): {miou_uninpainted:.4f}")
        print(f"    mIoU (Inpainted):   {miou_inpainted:.4f}")
        print("-" * 20)


def plot_sam_results(
        cfg,
        results_dataset,
        eval_dataset,
        fig_n: str,
        patch_index: int = None,
        iou_threshold: float = 0.5,
        output_dir: str = None
        ):
    """
    Visualize SAM vs ground truth for a single patch.

    Input:
        cfg : DictConfig / Namespace
            The Hydra config loaded from sam_config.yaml. Must contain
            general.infra_group, and infra group nodes (masonry_bridges / tunnel).
        results_dataset : SAMResultsDataset
            Pre-instantiated results dataset (with SAM outputs stored in .npz files).
        eval_dataset : EvaluationDataset
            Pre-instantiated evaluation dataset containing original/inpainted depth.
        fig_n : str
            Figure name (used as output filename base).
        patch_index : int | None
            Index of the patch to visualize. If None, uses cfg.general.patch_index.
        iou_threshold : float
            Threshold to decide whether to overlay predicted masks.
        output_dir : str | None
            Directory to save the visualization. If None, defaults to
            ./sam_vis/{infra_group}/
    """
    # Resolve patch index & output directory
    infra_group = cfg.general.infra_group
    if output_dir is None:
        output_dir = os.path.join('sam_vis', infra_group)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Generating SAM Visualization for Patch {patch_index} ({infra_group}) =====")

    # Optional: compute and print mIoU statistics once per call
    calculate_and_print_miou(results_dataset)

    sam_data = results_dataset.get_patch_by_index(
        patch_index
        )
    print(sam_data['path'])
    image_data = eval_dataset.get_patch_by_index(
        patch_index
        )
    print(image_data['original_depth'].shape)
    if not sam_data or not image_data:
        print(f"Could not find all necessary data for patch {patch_index}. "
              f"Skipping plot.")
        return

    sam_results = sam_data['sam_results']
    uninpainted_img = image_data['original_depth']
    inpainted_img = image_data['inpainted_depth']
    mask_inpainting = image_data['mask_inpainting']
    mask_boundary = image_data['mask_boundary']
    # Normalize for consistent visualization
    uninpainted_norm = norm_depth_unstd(
                uninpainted_img, 
                mask_inpainting=mask_inpainting, 
                mask_boundary=mask_boundary,
                abs_normalize=True
                )
    uninpainted_norm = np.mean(
                uninpainted_norm, 
                axis=2
                )
    inpainted_norm = inpainted_img.copy()  # Already normalized to [0, 1]
    # Grey scale

    inpainted_norm = np.mean(
        inpainted_norm, 
        axis=2
        )

    fig, axs = plt.subplots(1, 4, figsize=(28, 7))

    axs[0].imshow(
        inpainted_img, 
        cmap='gray'
        )
    axs[1].imshow(
        inpainted_img, 
        cmap='gray'
        )
    axs[2].imshow(
        uninpainted_norm, 
        cmap='gray'
        )
    axs[3].imshow(
        uninpainted_norm, 
        cmap='gray'
        )

    num_labels = sum(
        1 for key in sam_results
        if key.startswith('gt_mask')
        )
    colors = plt.get_cmap(
                'nipy_spectral'
                )(np.linspace(0.1, 0.9, num_labels))
            

    for label_idx in range(num_labels):
        color = list(colors[label_idx]) 
        color[3] = 0.25
        
        show_mask(
            sam_results[f'gt_mask_{label_idx}'], 
            axs[0], 
            color=color
            )
        if sam_results[f'iou_inpainted_{label_idx}'] > iou_threshold: 
            show_mask(
                sam_results[f'pred_mask_inpainted_{label_idx}'], 
                axs[1], 
                color=color
                )
            show_mask(
                sam_results[f'pred_mask_uninpainted_{label_idx}'], 
                axs[3], 
                color=color
                )
        # draw prompt coordinates
        for ax in [axs[1], axs[3]]:
            pos_prompt = sam_results[f'positive_prompt_{label_idx}']
            ax.scatter(
                pos_prompt[:, 0], 
                pos_prompt[:, 1], 
                color='green', 
                marker='*', 
                s=200, 
                edgecolor='white'
                )
            neg_prompt = sam_results[f'negative_prompt_{label_idx}']
            if neg_prompt is not None and neg_prompt.ndim > 0:
                continue

    for ax in axs.ravel(): 
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(output_dir, f"{fig_n}.jpg")
    plt.savefig(
        save_path, 
        bbox_inches='tight', 
        pad_inches=0.1
        )
    plt.close(fig)
    print(f"  Saved final visualization to {save_path}")