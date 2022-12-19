import torch 
from torch import nn

import random

class FairBCELoss(nn.Module):
    """
    ラベルの数が同数になるように損失を計算する Binary Cross Entropy
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
    def forward(self, outputs, labels, attention_mask, device):
        y = self.bce(outputs, labels) # Same shape as input.
        loss_mask = self.generate_loss_mask(labels, attention_mask).to(device)
        zero_cuda = torch.tensor(0, dtype=y.dtype).to(device)
        y = torch.where(loss_mask==1, y, zero_cuda) # Mask
        loss = torch.sum(y)
        return loss

    def generate_loss_mask(self, labels, attention_mask):
        loss_mask = []
        for label, a_m in zip(labels.tolist(), attention_mask.tolist()):  
            err_idx = [i for i, x in enumerate(label) if x == 1]      
            non_err_idx = [i for i, x in enumerate(zip(label, a_m)) if x[0]==0 and x[1]!=0]
            selected = random.sample(non_err_idx, len(err_idx))
            l_m = [0]*len(label)
            for i in err_idx+selected:
                l_m[i] = 1
            loss_mask.append(l_m)
        return torch.tensor(loss_mask)
              



