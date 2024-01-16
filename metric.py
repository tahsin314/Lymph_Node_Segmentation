import numpy as np
import torch

def dice_coefficient(predictions, targets, epsilon=1e-7):
    # Apply argmax to convert two-channel masks to a single-channel mask
    predicted_masks = np.argmax(predictions, axis=1)
    target_masks = np.argmax(targets, axis=1)
    # Calculate the Dice score for the single-channel masks
    intersection = np.sum(predicted_masks * target_masks, axis=(1, 2))
    union = np.sum(predicted_masks, axis=(1, 2)) + np.sum(target_masks, axis=(1, 2))
    dice_scores = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice_scores


def dice_coefficient_one_class(pred, target, threshold=0.5, smooth=1e-6):
    if threshold is not None:
        pred = (pred > threshold).astype(float)  # FocusNet
    # print(np.unique(pred), np.unique(target))
    num = pred.shape[0]
    m1 = pred.reshape(num, -1)  # Flatten
    m2 = target.reshape(num, -1)  # Flatten
    intersection = (m1 * m2).sum(axis=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(axis=1) + m2.sum(axis=1) + smooth)
    # print(np.mean(dice_score))
    return dice_score

def dice_score_by_data_torch(target, pred, threshold=0.5, smooth = 1e-6):
    # Calculating PCS threshold
    thr_pcs = np.log(1/threshold - 1 + smooth)
    num = pred.size(0)
    # Probability Correction Strategy --> Shallow Attention Network for Polyp Segmentation
    pred[torch.where(pred > thr_pcs)] /= (pred > thr_pcs).float().mean()
    pred[torch.where(pred < thr_pcs)] /= (pred < thr_pcs).float().mean()
    pred = pred.sigmoid()
    if threshold is not None:
        pred = (pred > threshold).float() 
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
    return dice_score

def recall(y_true, y_pred, threshold=0.5, smooth=1e-6):
    num_samples = y_true.size(0)
    # Calculating PCS threshold
    thr_pcs = np.log(1/threshold - 1 + smooth)
    num = y_pred.size(0)
    # Probability Correction Strategy --> Shallow Attention Network for Polyp Segmentation
    y_pred[torch.where(y_pred > thr_pcs)] /= (y_pred > thr_pcs).float().mean()
    y_pred[torch.where(y_pred < thr_pcs)] /= (y_pred < thr_pcs).float().mean()
    y_pred = y_pred.sigmoid()
    if threshold is not None:
        y_pred = (y_pred > threshold).float()
    y_true_f = y_true.view(num_samples, -1)
    y_pred_f = y_pred.view(num_samples, -1)

    # true positives, false positives, false negatives
    tp = (y_true_f * y_pred_f).sum(dim=1)
    fn = (y_true_f * (1 - y_pred_f)).sum(dim=1)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall
    
if __name__ == '__main__':
    a = np.random.randn()




