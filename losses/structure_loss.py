import torch
import torch.nn.functional as F

def structure_loss(mask, pred, kernel_size=37, stride=1, padding=18,  alpha=1, beta=2, smooth=1e-5):
    pred = torch.squeeze(pred)
    mask = torch.squeeze(mask)
    mask = mask.to(dtype=torch.float32)
    weit = 1 + 20*torch.abs(F.avg_pool2d(mask, kernel_size=kernel_size, stride=stride, padding=padding) - mask)
    pred = pred.sigmoid()
    intersection = ((pred * mask)*weit).sum(dim=(1, 2))
    union = ((pred + mask)*weit).sum(dim=(1, 2))
    logwiou =  -torch.log((intersection + smooth)/(union - intersection + smooth))
    logwdice = -torch.log((2. * intersection + smooth) / (union + smooth))
    return alpha*logwiou.mean() + beta*logwdice.mean()

def structure_loss_focusnet(mask, pred, kernel_size=31, stride=1, padding=15, alpha=1, beta=2, smooth=1e-5):
    pred = torch.squeeze(pred)
    mask = torch.squeeze(mask)
    mask = mask.to(dtype=torch.float32)
    avg_pooling = torch.abs(F.avg_pool2d(mask, kernel_size=kernel_size, stride=stride, padding=padding) - mask)
    neg_part_base = 1
    
    #omitting
    weit =  neg_part_base + 5*avg_pooling                                                   
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*bce)
    wbce = wbce.sum(dim=(1, 2))/weit.sum(dim=(1, 2))
    
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(1, 2))
    union = ((pred + mask)*weit).sum(dim=(1, 2))
    wiou = 1 - ((inter + 1)/(union - inter+1))
    
    m_wbce = wbce.mean()
    m_iou = wiou.mean()

    return alpha*m_wbce + beta*m_iou

def total_structure_loss(mask, preds, kernel_size=37, stride=1, padding=18,  alpha=1, beta=2, smooth=1e-5):
    loss = 0
    for pred in preds:
        loss += structure_loss(mask, pred, kernel_size, stride, padding,  alpha, beta, smooth)
    return loss/len(preds)

def total_structure_loss_focusnet(mask, preds, kernel_size=31, stride=1, padding=15, alpha=1, beta=2, smooth=1e-5):
    loss = 0
    for pred in preds:
        loss += structure_loss_focusnet(mask, pred, kernel_size, stride, padding,  alpha, beta, smooth)
    return loss/len(preds)
