###################### Post Processing ######################
import numpy as np
from skimage.morphology import remove_small_objects, binary_opening, disk


def post_process_predictions(predictions, class_id, min_object_size=5, apply_opening=True, kernel_size=3):
    """
    Post-process predictions for a specific class
    
    Args:
        predictions: Multi-class prediction masks
        class_id: Class to post-process (1=Ventricles, 2=Abnormal WMH)
        min_object_size: Minimum object size in pixels
        apply_opening: Whether to apply morphological opening
        kernel_size: Size of morphological kernel
    
    Returns:
        post_processed: Cleaned binary masks for the specific class
    """
    post_processed = np.zeros_like(predictions, dtype=np.uint8)
    
    for i in range(predictions.shape[0]):
        # Extract binary mask for specific class
        mask = (predictions[i] == class_id).astype(bool)
        
        # Remove small objects
        if min_object_size > 0:
            mask = remove_small_objects(mask, min_size=min_object_size)
        
        # Apply morphological opening
        if apply_opening:
            kernel = disk(kernel_size)
            mask = binary_opening(mask, kernel)
            if min_object_size > 0:
                mask = remove_small_objects(mask, min_size=min_object_size)
        
        post_processed[i] = mask.astype(np.uint8)
    
    return post_processed
