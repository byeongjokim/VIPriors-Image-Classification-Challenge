import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1]).cuda()
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        
        return cosine_loss + self.xent * cent_loss

class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1, reduction="mean"):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1]).cuda()
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        
        return cosine_loss + self.xent * focal_loss
