import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet

import torch.nn.functional as F



def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        predicted = predicted.sigmoid()
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss
    
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # print(predict.size(), target.size(), predict.max(), target.max())
        # predict = predict.sigmoid()
        predict = F.softmax(predict, dim=1)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predicted, target):
        # Flatten the inputs
        predicted = predicted.sigmoid()
        predicted = predicted.view(-1)
        target = target.view(-1)

        # True Positives, False Positives, False Negatives
        tp = (predicted * target).sum()
        fp = ((1 - target) * predicted).sum()
        fn = (target * (1 - predicted)).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Tversky loss
        tversky_loss = 1 - tversky

        return tversky_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predicted, target):
        # Flatten the inputs
        predicted = predicted.sigmoid()
        predicted = predicted.view(-1)
        target = target.view(-1)

        # Compute the binary cross entropy loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(predicted, target.float(), reduction='none')

        # Calculate the modulating factor
        p = torch.exp(-bce_loss)
        modulating_factor = (1 - p) ** self.gamma

        # Calculate the focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocusNetLoss(nn.Module):
    def __init__(self, alpha=2, beta=1):
        super(FocusNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()

    def forward(self, preds, targets):
        fl = self.focal_loss(preds, targets)
        tl = self.tversky_loss(preds, targets)
        print(f"Focal: {fl.detach().cpu().numpy():.3f} Tversky: {tl.detach().cpu().numpy():.3f}")
        return self.alpha*fl + self.beta*tl


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.9, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    """ 
    y_pred ---> unnormalized predicted logits: B x num_class x H x W
    y_true ---> ground_truth labels: B x 1 x H x W 
                where 0 <= y_target[:,:,:] <= num_class - 1 
    """
    def forward(self, y_pred, y_true):
        self.epsilon = torch.finfo(y_pred.dtype).eps    #can use pre-defined too
        P = F.softmax(y_pred, dim=1)
        # Clip values to prevent division by zero error
        P = torch.clamp(P, self.epsilon, 1. - self.epsilon)

        # Make Target one-hot vector: y_true --> class_mask:B x num_class x H x W
        class_mask = torch.zeros(P.shape).to(P.device)
        class_mask.scatter_(1, y_true.long(), 1.)

        axis = identify_axis(class_mask.shape)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(class_mask * P, dim=axis)
        fn = torch.sum(class_mask * (1-P), dim=axis)
        fp = torch.sum((1-class_mask) * P, dim=axis)

        dice_class = (tp + self.epsilon)/(tp + self.delta*fp + (1-self.delta)*fn + self.epsilon)
        # print('dice_class: ',dice_class.size())

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = torch.pow(1-dice_class[:,1], 1-self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], dim=-1))
        return loss

class AsymmetricFocalLoss(nn.Module):
    """
    This is the implementation for binary segmentation.
    For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
    gamma : float, optional
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.9, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    # y_pred ---> unnormalized predicted logits: B x num_class x H x W
    # y_true ---> ground_truth labels: B x 1 x H x W where 0 <= y_target[:,:,:,:] <= num_class - 1
    def forward(self, y_pred, y_true):

        self.epsilon = torch.finfo(y_pred.dtype).eps    #can use pre-defined too
        P = F.softmax(y_pred, dim = 1)
        P = torch.clamp(P, self.epsilon, 1. - self.epsilon)
        # log_P = F.log_softmax(y_pred, dim=1)
        log_P = torch.log(P)

        # Make Target one-hot vector: y_true --> class_mask:B x num_class x H x W
        class_mask = torch.zeros(P.shape).to(P.device)
        class_mask.scatter_(1, y_true.long(), 1.)
        cross_entropy = -class_mask * log_P # cross_entropy = -class_mask*torch.log(P) ok too
        
	    # Calculate losses separately for each class, only suppressing background class
        # gamma>>1 --> more punishing
        back_ce = torch.pow(1 - P[:,0,:,:], self.gamma) * cross_entropy[:,0,:,:]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,1,:,:]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha=1, beta=2):
        super(UnifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = AsymmetricFocalLoss(delta=0.9, gamma=3)
        self.tversky_loss = AsymmetricFocalTverskyLoss(delta=0.9, gamma=3)

    def forward(self, preds, targets):
        fl = self.focal_loss(preds, targets)
        tl = self.tversky_loss(preds, targets)
        return self.alpha*fl + self.beta*tl


class CustomLoss(nn.Module):
    
    def __init__(self, alpha=0.3, beta=0.75, gamma=0.75, epsilon=1e-6):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def _tversky_index(self, predicted, targets):
        TP = torch.sum(predicted * targets, (1,2,3))
        FP = torch.sum((1. - targets) * predicted, (1,2,3))
        FN = torch.sum((1. - predicted) * targets, (1,2,3))
        return TP/(TP + self.alpha * FP + self.beta * FN + self.epsilon)

    def forward(self, predicted, targets):
        return torch.mean(torch.pow(1 - self._tversky_index(predicted, targets, self.alpha, self.beta, self.epsilon), self.gamma))




class StructureLoss(nn.Module):
    def __init__(self, cuda=True, device_ids = None):
        super(StructureLoss, self).__init__()
        self.cuda = cuda
        self.device_ids = device_ids   

    # From PraNet: https://arxiv.org/pdf/2006.11392.pdf
    # Repo: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py
    def forward(self, pred, mask):
        avg_pooling = F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7)
        neg_part_base = 1
        
        #omitting
        weit =  neg_part_base + 20*torch.abs(avg_pooling - mask)   
                                                            
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit*bce)
        wbce = wbce.sum(dim=(1, 2, 3))/weit.sum(dim=(1, 2, 3))
        
        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(1, 2, 3))
        union = ((pred + mask)*weit).sum(dim=(1, 2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        
        m_wbce = wbce.mean()
        m_iou = wiou.mean()

        return m_wbce + m_iou