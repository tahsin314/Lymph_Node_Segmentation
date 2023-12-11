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

def total_structure_loss(mask, preds):
    loss = 0
    for pred in preds:
        loss += structure_loss(mask, pred)
    return loss/len(preds)
