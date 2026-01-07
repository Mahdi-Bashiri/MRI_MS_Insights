###################### Extended Loss Functions for Brain FLAIR Segmentation ######################
###################### Libraries ######################
from keras import backend as K
import tensorflow as tf

###################### Basic Loss Functions ######################

def weighted_binary_crossentropy(pos_weight=1.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)  
        
        # Calculate weighted cross-entropy
        loss_pos = pos_weight * y_true * K.log(y_pred)
        loss_neg = (1 - y_true) * K.log(1 - y_pred)
        
        return -K.mean(loss_pos + loss_neg)
    return loss

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        focal_loss = -alpha_t * K.pow(1 - pt, gamma) * K.log(pt)
        return K.mean(focal_loss)
    return loss

def dice_loss():
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coef
    return loss

def combined_loss(alpha=0.5, pos_weight=100.0):
    def loss(y_true, y_pred):
        bce = weighted_binary_crossentropy(pos_weight)(y_true, y_pred)
        dice = dice_loss()(y_true, y_pred)
        return alpha * bce + (1 - alpha) * dice
    return loss

###################### Multi-Class Loss Functions ######################

def weighted_categorical_crossentropy(class_weights):
    """Standard weighted categorical crossentropy"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.constant(class_weights, dtype=tf.float32)
        class_weights_tensor = tf.gather(weights, K.argmax(y_true, axis=-1))
        cross_entropy = -K.sum(y_true * K.log(y_pred), axis=-1)
        weighted_loss = cross_entropy * class_weights_tensor
        return K.mean(weighted_loss)
    return loss

def multiclass_dice_loss(num_classes=3, class_weights=None):
    """Enhanced dice loss with optional class weights"""
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        dice_scores = []

        # Convert class_weights to tensor if provided
        if class_weights is not None:
            weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            # Apply class weights if provided
            if class_weights is not None:
                dice_coef = dice_coef * weights_tensor[class_idx]

            dice_scores.append(dice_coef)

        mean_dice = K.mean(K.stack(dice_scores))
        return 1 - mean_dice
    return loss

###################### Advanced Loss Functions ######################

def unified_focal_loss(class_weights, delta=0.6, gamma=0.5):
    """
    Unified Focal Loss - State-of-the-art for medical segmentation
    
    Combines Dice coefficient with precision-recall focal weighting.
    Best for imbalanced multi-class segmentation with small structures.
    
    Args:
        class_weights: Array of weights for each class [background, ventricles, WMH]
        delta: Weight for precision-recall component (0-1)
        gamma: Focusing parameter for Dice component (0-2)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        num_classes = K.int_shape(y_pred)[-1]

        # Convert class_weights to tensor once
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        unified_losses = []
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            # Calculate precision and recall
            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            tp = K.sum(y_true_f * y_pred_f)
            fp = K.sum((1 - y_true_f) * y_pred_f)
            fn = K.sum(y_true_f * (1 - y_pred_f))

            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)

            # Dice coefficient
            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

            # Unified focal loss: focuses on hard examples and boundary regions
            unified_loss = K.pow(1 - dice, gamma) * K.pow(1 - precision * recall, delta)

            # Apply class weights
            if class_weights is not None:
                unified_loss = unified_loss * weights_tensor[class_idx]

            unified_losses.append(unified_loss)

        return K.mean(K.stack(unified_losses))

    return loss


def tversky_loss(class_weights, alpha=0.7, beta=0.3):
    """
    Multi-class Tversky loss - Excellent for small structures like ventricles
    
    Generalizes Dice loss with controllable false positive/negative trade-off.
    Higher alpha penalizes false negatives more (better recall).
    Higher beta penalizes false positives more (better precision).
    
    Args:
        class_weights: Array of weights for each class
        alpha: Weight for false negatives (0-1), typically 0.7
        beta: Weight for false positives (0-1), typically 0.3
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        num_classes = K.int_shape(y_pred)[-1]

        # Convert class_weights to tensor once
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        tversky_scores = []
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            true_pos = K.sum(y_true_f * y_pred_f)
            false_neg = K.sum(y_true_f * (1 - y_pred_f))
            false_pos = K.sum((1 - y_true_f) * y_pred_f)

            tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

            # Apply class weights
            if class_weights is not None:
                tversky = tversky * weights_tensor[class_idx]

            tversky_scores.append(tversky)

        mean_tversky = K.mean(K.stack(tversky_scores))
        return 1 - mean_tversky

    return loss


def combined_wce_dice_loss(class_weights, wce_weight=0.5, dice_weight=0.5):
    """
    Combine weighted categorical crossentropy and dice losses
    
    Balances pixel-wise classification (CE) with region overlap (Dice).
    Good baseline for medical segmentation.
    
    Args:
        class_weights: Array of weights for each class
        wce_weight: Weight for cross-entropy component (0-1)
        dice_weight: Weight for dice component (0-1)
    
    Returns:
        Combined loss function
    """
    wce_loss_fn = weighted_categorical_crossentropy(class_weights)
    dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)

    def loss(y_true, y_pred):
        wce_loss = wce_loss_fn(y_true, y_pred)
        dice_loss = dice_loss_fn(y_true, y_pred)
        return wce_weight * wce_loss + dice_weight * dice_loss

    return loss


def multiclass_focal_loss(class_weights, alpha=0.25, gamma=2.0):
    """
    Multi-class focal loss for handling hard examples
    
    Focuses training on hard-to-classify pixels.
    Good for extreme class imbalance.
    
    Args:
        class_weights: Array of weights for each class
        alpha: Balance parameter (0-1)
        gamma: Focusing parameter, typically 2.0
    
    Returns:
        Focal loss function
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate focal loss
        cross_entropy = -y_true * K.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        focal_loss = focal_weight * cross_entropy

        # Apply class weights
        weights = tf.constant(class_weights, dtype=tf.float32)
        class_weights_tensor = tf.gather(weights, K.argmax(y_true, axis=-1))
        class_weights_expanded = tf.expand_dims(class_weights_tensor, axis=-1)

        weighted_focal_loss = focal_loss * class_weights_expanded
        return K.mean(K.sum(weighted_focal_loss, axis=-1))

    return loss


def exponential_logarithmic_loss(class_weights, gamma_dice=0.3, gamma_cross=0.3):
    """
    Exponential Logarithmic Loss - Good for boundary refinement
    
    Non-linear combination of Dice and CE for better gradient flow.
    Helps with precise boundary delineation.
    
    Args:
        class_weights: Array of weights for each class
        gamma_dice: Exponent for dice component
        gamma_cross: Exponent for cross-entropy component
    
    Returns:
        Exponential-logarithmic loss function
    """
    def loss(y_true, y_pred):
        # Dice component
        dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)
        dice_loss = dice_loss_fn(y_true, y_pred)

        # Cross entropy component
        ce_loss_fn = weighted_categorical_crossentropy(class_weights)
        ce_loss = ce_loss_fn(y_true, y_pred)

        # Exponential logarithmic formulation
        exp_log_loss = (K.pow(-K.log(1 - dice_loss + K.epsilon()), gamma_dice) +
                        K.pow(ce_loss, gamma_cross))

        return exp_log_loss

    return loss


def combo_loss(class_weights, ce_weight=0.5, dice_weight=0.5):
    """
    Simplified combo loss for stable training
    
    Balances cross-entropy and Dice with fixed weights.
    More stable than adaptive versions.
    
    Args:
        class_weights: Array of weights for each class
        ce_weight: Weight for cross-entropy (0-1)
        dice_weight: Weight for dice (0-1)
    
    Returns:
        Combined loss function
    """
    def loss(y_true, y_pred):
        # Cross entropy component
        ce_loss_fn = weighted_categorical_crossentropy(class_weights)
        ce_loss = ce_loss_fn(y_true, y_pred)

        # Dice loss component
        dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)
        dice_loss = dice_loss_fn(y_true, y_pred)

        combo = ce_weight * ce_loss + dice_weight * dice_loss
        return combo

    return loss


###################### Loss Function Recommendations ######################

def get_loss_function_info():
    """
    Get information about available loss functions and their use cases
    
    Returns:
        Dictionary with loss function descriptions and recommended parameters
    """
    return {
        'weighted_categorical': {
            'description': 'Standard weighted cross-entropy for multi-class',
            'best_for': 'Baseline, stable training',
            'parameters': 'class_weights only',
            'pros': 'Fast, stable, well-tested',
            'cons': 'Ignores spatial structure'
        },
        'unified_focal': {
            'description': 'State-of-the-art combining Dice and focal weighting',
            'best_for': 'Imbalanced classes, small structures (ventricles)',
            'parameters': 'class_weights, delta=0.6, gamma=0.5',
            'pros': 'Best for small structures, handles imbalance well',
            'cons': 'Slightly slower, needs tuning'
        },
        'tversky': {
            'description': 'Generalized Dice with FP/FN trade-off control',
            'best_for': 'Small structures, when recall > precision',
            'parameters': 'class_weights, alpha=0.7, beta=0.3',
            'pros': 'Excellent for tiny structures like ventricles',
            'cons': 'Needs alpha/beta tuning'
        },
        'combined_wce_dice': {
            'description': 'Balanced combination of CE and Dice',
            'best_for': 'General purpose, stable training',
            'parameters': 'class_weights, wce_weight=0.4, dice_weight=0.6',
            'pros': 'Good balance, stable',
            'cons': 'May not excel at any specific task'
        },
        'multiclass_focal': {
            'description': 'Focal loss for hard example mining',
            'best_for': 'Extreme imbalance, hard boundaries',
            'parameters': 'class_weights, alpha=0.25, gamma=2.0',
            'pros': 'Handles hard examples well',
            'cons': 'Can be unstable early in training'
        },
        'exponential_logarithmic': {
            'description': 'Non-linear combination for boundary refinement',
            'best_for': 'Precise boundaries, fine details',
            'parameters': 'class_weights, gamma_dice=0.3, gamma_cross=0.3',
            'pros': 'Excellent boundary accuracy',
            'cons': 'Complex, needs careful tuning'
        }
    }


def print_loss_function_guide():
    """Print a guide for choosing loss functions"""
    info = get_loss_function_info()
    
    print("\n" + "="*80)
    print("LOSS FUNCTION SELECTION GUIDE")
    print("="*80)
    
    for name, details in info.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {details['description']}")
        print(f"  Best for: {details['best_for']}")
        print(f"  Parameters: {details['parameters']}")
        print(f"  Pros: {details['pros']}")
        print(f"  Cons: {details['cons']}")
    
    print("\n" + "="*80)
    print("RECOMMENDED FOR YOUR TASK (Ventricles + WMH):")
    print("="*80)
    print("1. unified_focal - Best overall for imbalanced multi-class with small structures")
    print("2. tversky - Excellent specifically for small ventricles")
    print("3. combined_wce_dice - Safe baseline if others don't converge")
    print("="*80 + "\n")