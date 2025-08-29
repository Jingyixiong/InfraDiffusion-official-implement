import os

import numpy as np
import torch
import cv2

from typing import Optional
from segment_anything import sam_model_registry, SamPredictor

class SegmentAnythingModel:
    """
    A wrapper class for the Segment Anything Model (SAM) to provide a simple
    interface for point-prompted segmentation.
    """
    def __init__(
            self,
            model_type: str,
            checkpoint_path: str
            ):

        if not os.path.exists(
            checkpoint_path
            ):
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {checkpoint_path}"
                )
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully.")

    def predict(
            self, 
            image_np: np.ndarray, 
            positive_points: np.ndarray, 
            negative_points: Optional[np.ndarray] = None
            ) -> np.ndarray:
        """
            Runs SAM prediction on a single image with positive 
            and optional negative point prompts.
        """
        image_uint8 = (image_np * 255).astype(np.uint8)
         # If the image has a single channel dimension (H, W, 1), squeeze it to (H, W)
        if image_uint8.ndim == 3 and image_uint8.shape[2] == 1:
            image_uint8 = np.squeeze(image_uint8, axis=2)
        
        # Check if the image is 2D (grayscale) and convert to RGB if needed
        if image_uint8.ndim == 2:
            # It's grayscale, convert to RGB
            image_rgb = cv2.cvtColor(
                image_uint8, 
                cv2.COLOR_GRAY2RGB
                )
        elif image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
            # It's already in the correct format
            image_rgb = image_uint8
        else:
            raise ValueError(
                f"Unsupported image format with shape: {image_np.shape}"
                )
        self.predictor.set_image(image_rgb)
        
        if negative_points is not None and len(negative_points) > 0:
            input_points = np.concatenate(
                [positive_points, negative_points], axis=0
            )
            input_labels = np.array(
                [1] * len(positive_points) + [0] * len(negative_points)
            )
        else:
            input_points = positive_points
            input_labels = np.array([1] * len(positive_points))

        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)

