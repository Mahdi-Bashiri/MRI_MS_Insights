###################### Metrics and Evaluation ######################

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


def calculate_class_weights(masks, num_classes):
    """Calculate class weights inversely proportional to class frequency"""
    flattened = masks.flatten()
    class_counts = np.bincount(flattened, minlength=num_classes)
    total_pixels = len(flattened)
    class_weights = total_pixels / (num_classes * class_counts)
    class_weights = class_weights / class_weights[0]
    return class_weights

def dice_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate Dice coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    smooth = 1e-6
    intersection = np.sum(y_true_class * y_pred_class)
    return (2. * intersection + smooth) / (np.sum(y_true_class) + np.sum(y_pred_class) + smooth)

def iou_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate IoU (Intersection over Union) coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
    
    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_surface_distances(mask_gt, mask_pred):
    """
    Compute surface distances using proper erosion-based surface extraction.
    Returns distances from pred surface to GT surface and vice versa.
    """
    # If either mask is empty, return None
    if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return None, None
    
    # Extract surface using erosion (surface = original - eroded)
    surface_gt = mask_gt.astype(bool) & ~binary_erosion(mask_gt.astype(bool))
    surface_pred = mask_pred.astype(bool) & ~binary_erosion(mask_pred.astype(bool))
    
    # If no surface points found, return None
    if np.sum(surface_gt) == 0 or np.sum(surface_pred) == 0:
        return None, None
    
    # Get coordinates of surface points
    coords_gt = np.argwhere(surface_gt)
    coords_pred = np.argwhere(surface_pred)
    
    # Compute minimum distances from each pred surface point to GT surface
    distances_pred_to_gt = np.min(cdist(coords_pred, coords_gt, metric='euclidean'), axis=1)
    
    # Compute minimum distances from each GT surface point to pred surface
    distances_gt_to_pred = np.min(cdist(coords_gt, coords_pred, metric='euclidean'), axis=1)
    
    return distances_pred_to_gt, distances_gt_to_pred

def hausdorff_distance_95(y_true, y_pred):
    """
    Calculate 95th percentile Hausdorff Distance (HD95).
    
    HD95 measures the 95th percentile of surface distances, making it robust to outliers.
    Returns distance in pixels.
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
    
    Returns:
        hd95: 95th percentile Hausdorff distance in pixels, or np.inf if masks are empty
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty masks
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0  # Perfect match (both empty)
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return np.inf  # One empty, one not - worst case
    
    # Get surface distances
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(y_true, y_pred)
    
    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.inf
    
    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    
    # Calculate 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return hd95

def average_symmetric_surface_distance(y_true, y_pred):
    """
    Calculate Average Symmetric Surface Distance (ASSD).
    
    ASSD measures the average distance between surfaces of prediction and ground truth,
    providing an evaluation of the segmentation's boundary accuracy.
    Returns distance in pixels.
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
    
    Returns:
        assd: Average symmetric surface distance in pixels, or np.inf if masks are empty
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty masks
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0  # Perfect match (both empty)
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return np.inf  # One empty, one not - worst case
    
    # Get surface distances
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(y_true, y_pred)
    
    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.inf
    
    # Calculate average of all surface distances (symmetric)
    assd = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2.0
    
    return assd

###################### Metrics Calculation ######################

def calculate_class_metrics(y_true, y_pred, class_id, class_name):
    """
    Calculate metrics for a specific class
    
    Args:
        y_true: Ground truth masks (H, W) with class labels
        y_pred: Predicted masks (H, W) with class labels
        class_id: ID of the class (1 or 2)
        class_name: Name of the class for printing
    
    Returns:
        dict: Dictionary containing all metrics for this class
    """
    # Extract binary masks for this class
    y_true_binary = (y_true == class_id).astype(np.uint8)
    y_pred_binary = (y_pred == class_id).astype(np.uint8)
    
    # Flatten for sklearn metrics
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Calculate Dice
    dice = dice_coefficient_multiclass(y_true, y_pred, class_id)
    
    # Calculate IoU
    iou = iou_coefficient_multiclass(y_true, y_pred, class_id)
    
    # Calculate HD95
    hd95 = hausdorff_distance_95(y_true_binary, y_pred_binary)
    
    metrics = {
        f'Precision_{class_name}': precision,
        f'Recall_{class_name}': recall,
        f'Dice_{class_name}': dice,
        f'IoU_{class_name}': iou,  # ADDED
        f'HD95_{class_name}': hd95
    }
    
    print(f"\n{class_name} Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Dice: {dice:.4f}")
    print(f"  IoU: {iou:.4f}")  # ADDED
    print(f"  HD95: {hd95:.4f}")
    
    return metrics

def calculate_per_image_metrics(y_true_images, y_pred_images, class_id):
    """
    Calculate per-image metrics for statistical analysis
    
    Args:
        y_true_images: Ground truth masks array (N, H, W)
        y_pred_images: Predicted masks array (N, H, W)
        class_id: Class ID to evaluate
    
    Returns:
        Dictionary with arrays of per-image metrics
    """
    dice_scores = []
    iou_scores = []
    hd95_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(y_true_images)):
        y_true_binary = (y_true_images[i] == class_id).astype(np.uint8)
        y_pred_binary = (y_pred_images[i] == class_id).astype(np.uint8)
        
        # Dice
        dice = dice_coefficient_multiclass(y_true_images[i], y_pred_images[i], class_id)
        dice_scores.append(dice)
        
        # IoU
        iou = iou_coefficient_multiclass(y_true_images[i], y_pred_images[i], class_id)
        iou_scores.append(iou)
        
        # HD95
        hd95 = hausdorff_distance_95(y_true_binary, y_pred_binary)
        hd95_scores.append(hd95)
        
        # Precision and Recall
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        precision_scores.append(prec)
        recall_scores.append(rec)
    
    return {
        'dice': np.array(dice_scores),
        'iou': np.array(iou_scores),  # ADDED
        'hd95': np.array(hd95_scores),
        'precision': np.array(precision_scores),
        'recall': np.array(recall_scores)
    }

def calculate_overall_metrics(y_true, y_pred):
    """
    Calculate overall metrics (macro-average across classes)
    
    Note: This function should NOT be used for HD95 calculation on flattened data.
    HD95 should be computed per-image and then averaged.
    """
    # Ventricles metrics (without HD95)
    y_true_vent = (y_true == 1).astype(np.uint8)
    y_pred_vent = (y_pred == 1).astype(np.uint8)
    
    y_true_vent_flat = y_true_vent.flatten()
    y_pred_vent_flat = y_pred_vent.flatten()
    
    precision_vent = precision_score(y_true_vent_flat, y_pred_vent_flat, zero_division=0)
    recall_vent = recall_score(y_true_vent_flat, y_pred_vent_flat, zero_division=0)
    dice_vent = dice_coefficient_multiclass(y_true, y_pred, 1)
    iou_vent = iou_coefficient_multiclass(y_true, y_pred, 1)
    
    # Abnormal WMH metrics (without HD95)
    y_true_wmh = (y_true == 2).astype(np.uint8)
    y_pred_wmh = (y_pred == 2).astype(np.uint8)
    
    y_true_wmh_flat = y_true_wmh.flatten()
    y_pred_wmh_flat = y_pred_wmh.flatten()
    
    precision_wmh = precision_score(y_true_wmh_flat, y_pred_wmh_flat, zero_division=0)
    recall_wmh = recall_score(y_true_wmh_flat, y_pred_wmh_flat, zero_division=0)
    dice_wmh = dice_coefficient_multiclass(y_true, y_pred, 2)
    iou_wmh = iou_coefficient_multiclass(y_true, y_pred, 2)
    
    # Create metrics dictionaries
    vent_metrics = {
        'Precision_Ventricles': precision_vent,
        'Recall_Ventricles': recall_vent,
        'Dice_Ventricles': dice_vent,
        'IoU_Ventricles': iou_vent,
        'HD95_Ventricles': 0.0  # Placeholder - will be computed per-image in main script
    }
    
    wmh_metrics = {
        'Precision_WMH': precision_wmh,
        'Recall_WMH': recall_wmh,
        'Dice_WMH': dice_wmh,
        'IoU_WMH': iou_wmh,
        'HD95_WMH': 0.0  # Placeholder - will be computed per-image in main script
    }
    
    # Overall (macro-average)
    overall_metrics = {
        'Precision_Overall': (precision_vent + precision_wmh) / 2,
        'Recall_Overall': (recall_vent + recall_wmh) / 2,
        'Dice_Overall': (dice_vent + dice_wmh) / 2,
        'IoU_Overall': (iou_vent + iou_wmh) / 2,
        'HD95_Overall': 0.0  # Placeholder - will be computed per-image in main script
    }
    
    print(f"\nVentricles Metrics:")
    print(f"  Precision: {precision_vent:.4f}")
    print(f"  Recall: {recall_vent:.4f}")
    print(f"  Dice: {dice_vent:.4f}")
    print(f"  IoU: {iou_vent:.4f}")
    
    print(f"\nWMH Metrics:")
    print(f"  Precision: {precision_wmh:.4f}")
    print(f"  Recall: {recall_wmh:.4f}")
    print(f"  Dice: {dice_wmh:.4f}")
    print(f"  IoU: {iou_wmh:.4f}")
    
    print(f"\nOverall Metrics (Macro-Average):")
    print(f"  Precision: {overall_metrics['Precision_Overall']:.4f}")
    print(f"  Recall: {overall_metrics['Recall_Overall']:.4f}")
    print(f"  Dice: {overall_metrics['Dice_Overall']:.4f}")
    print(f"  IoU: {overall_metrics['IoU_Overall']:.4f}")
    print(f"  HD95: (computed separately per-image)")
    
    # Combine all metrics
    all_metrics = {**vent_metrics, **wmh_metrics, **overall_metrics}
    
    return all_metrics

