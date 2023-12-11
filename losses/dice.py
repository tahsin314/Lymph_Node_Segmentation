import torch
import torch.nn.functional as F

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss function for segmentation.
    :param y_true: ground truth segmentation mask
    :param y_pred: predicted segmentation mask
    :param smooth: smoothing factor to avoid division by zero
    :return: Dice loss
    """
    # flatten inputs
    y_pred = y_pred.sigmoid()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    # true positives, false positives, false negatives
    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()

    # Dice coefficient
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    # print(dice.mean().detach().cpu().numpy())
    # Dice loss

    dice_loss = 1 - dice
    gamma = 0.75
    # return torch.pow(dice_loss, gamma)
    return dice_loss

def dice_lossv2(target, pred, smooth = 1e-4):
    num = pred.size(0)
    pred = pred.sigmoid()
    
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
    if torch.isnan(dice_score).any() or torch.isnan(torch.log(dice_score)).any():
        print('nan detected')
        print(pred.mean())
        print(intersection)
        print(m1.sum(), m2.sum())
    dice_loss = 1 - dice_score
    # dice_loss = -torch.log(dice_score)
    # print(dice_loss.mean())
    gamma = 0.75
    # return torch.pow(dice_loss, gamma).mean()
    return dice_loss.mean()