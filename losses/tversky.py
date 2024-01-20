import torch
import torch.nn.functional as F

def tversky(y_true, y_pred, alpha=0.7, beta=3, smooth=1e-6):
    """
    Tversky loss function for segmentation.
    :param y_true: ground truth segmentation mask
    :param y_pred: predicted segmentation mask
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param smooth: smoothing factor to avoid division by zero
    :return: Tversky loss for each element in y_pred
    """
    # Flatten inputs
    y_pred = y_pred.sigmoid()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    # weit = 1 + 20*torch.abs(F.avg_pool2d(y_true, kernel_size=37, stride=1, padding=18) - y_true)
    # True positives, false positives, false negatives
    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()

    # Tversky index
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    # Reshape the result to have the same shape as y_pred
    # tversky = tversky.view()

    return tversky



def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    loss = 1 - tversky(y_true, y_pred, alpha, beta, smooth)
    # loss = -torch.log(tversky(y_true, y_pred, alpha, beta, smooth))
    return loss
 
def focal_tversky(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    pt_1 = tversky(y_true, y_pred, alpha, beta, smooth)
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)