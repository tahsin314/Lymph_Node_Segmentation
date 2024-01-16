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

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class BinaryTverskyLoss(nn.Module):
    def __init__(self, delta = 0.7, gamma = 3, size_average=True):
        super(BinaryTverskyLoss, self).__init__()
        self.size_average = size_average
        # self.reduce = reduce
        self.delta = delta
        self.gamma = gamma
    # preds --> B x 1 x H x W
    # targets --> B x 1 x H x W where targets[:,:,:,:] = 0/1
    def forward(self, preds, targets):

        N = preds.size(0)
        P = F.sigmoid(preds)
        
        P = P.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        class_mask = targets.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        
        smooth = torch.zeros(1, dtype=torch.float32).fill_(0.00001)
        smooth = smooth.to(P.device)
        ones = torch.ones(P.shape).to(P.device)

        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask
        
        # self.beta = 1 - self.delta
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.delta * torch.sum(FP, dim=(0)).float() + (1-self.delta) * torch.sum(FN, dim=(0)).float()
        # print(num.size())
        # print(den.size())
        # num = TP
        # den = num + self.delta * FP + (1-self.delta) * FN

        TI = num / (den + smooth)

        if self.size_average:       
            ones = torch.ones(TI.shape).to(TI.device)
            loss = ones - TI.mean()
        else:
            TI = TI.sum()
            # ones = torch.ones(TI.shape).to(TI.device)
            loss = 1. - TI

        # loss = 1. - TI
        # asym_focal_tl = torch.where(torch.eq(class_mask, 1), loss.pow(1-self.gamma), loss)
        
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=3, size_average=False):
        super(BinaryFocalLoss, self).__init__()
        # if alpha is None:
        #     self.delta = torch.Tensor([0.7]).cuda()
        # else:
        #     self.delta = torch.Tensor([delta]).cuda()
        self.delta = delta
        self.gamma = gamma
        self.size_average = size_average

    # preds --> B x 1 x H x W
    # targets --> B x 1 x H x W where targets[:,:,:,:] = 0/1
    def forward(self, preds, targets):
        N = preds.size(0)

        # preds = preds.permute(0, 2, 3, 1).contiguous().view(-1)
        # targets = targets.permute(0, 2, 3, 1).contiguous().view(-1)
        # ##########################################################
        P = F.sigmoid(preds)
        log_P = F.logsigmoid(preds)
        log_P_ = F.logsigmoid(1 - preds)
        ############################################################

        #targets = targets.float()

        batch_loss = -self.delta * log_P * targets - (1 - self.delta) * P.pow(self.gamma)*log_P_ * (1-targets) 
        # print(batch_loss.size(), batch_loss.mean())
        
        if self.size_average:
            loss = batch_loss.mean()
            # print(loss, loss.size(), loss.mean())
        else:
            loss = torch.sum(batch_loss, dim=(1,2,3))
            # loss = batch_loss.sum()
        # print(loss, loss.size(), loss.mean())

        return loss.mean()

class FocusNetLoss(nn.Module):
    def __init__(self):
        super(FocusNetLoss, self).__init__()
        self.focal_loss = BinaryFocalLoss(delta=0.7, gamma=3, size_average=True)
        self.tversky_loss = BinaryTverskyLoss(delta=0.7, gamma=3, size_average=False)

    def forward(self, preds, targets):
        fl = self.focal_loss(preds, targets)
        tl = self.tversky_loss(preds, targets)
        return fl, tl

        
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        gpu = preds.get_device()
        device = torch.device('cuda:'+str(gpu))
        self.alpha = self.alpha.to(device)

        P = F.softmax(preds, dim=1)
        log_P = F.log_softmax(preds, dim=1)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.)

        alpha = self.alpha[targets.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_probs = (log_P * class_mask).sum(1).view(-1, 1)

        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs
        if weight is not False:
            element_weight = weight.squeeze(0)[targets.squeeze(0)]
            batch_loss = batch_loss * element_weight

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

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
    def __init__(self):
        super(UnifiedFocalLoss, self).__init__()
        self.focal_loss = AsymmetricFocalLoss(delta=0.9, gamma=3)
        self.tversky_loss = AsymmetricFocalTverskyLoss(delta=0.9, gamma=3)

    def forward(self, preds, targets):
        fl = self.focal_loss(preds, targets)
        tl = self.tversky_loss(preds, targets)
        return fl, tl


class CustomLoss(nn.Module):
    
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.alpha = 0.3
        self.beta = 0.7
        self.gamma = 0.75
        self.epsilon = 1e-6

    def _tversky_index(self, predicted, targets, alpha, beta, epsilon):
        TP = torch.sum(predicted * targets, (1,2,3))
        FP = torch.sum((1. - targets) * predicted, (1,2,3))
        FN = torch.sum((1. - predicted) * targets, (1,2,3))
        return TP/(TP + alpha * FP + beta * FN + epsilon)

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

        return m_wbce, m_iou