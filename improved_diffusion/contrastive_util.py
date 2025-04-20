import torch.nn as nn
import torch.nn.functional as F

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        
        self.l1 = nn.L1Loss()
        self.l2=nn.MSELoss(reduction='mean')  
        self.weights = [1.0/32.0, 1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0]
        self.ab = ablation
    def forward(self, features_N, features_P, features_A):
        loss = 0
        for i in range(len(features_N)):
            d_ap = self.l2(features_A[i], features_P[i].detach())  # L2(anchor, positive)
            if not self.ab:
                d_an = self.l2(features_A[i], features_N[i].detach())  # L2(anchor, negative)
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap
            loss += self.weights[i] * contrastive
        return loss
