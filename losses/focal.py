# source: losses.py --> 2d_model_setup
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, target, input):
        # Flatten the input and target tensors
        input = torch.sigmoid(input)
        input = input.view(-1)
        target = target.view(-1)

        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Calculate the focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        return torch.mean(focal_loss)
