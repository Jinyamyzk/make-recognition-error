from numpy import dtype
import torch 
from torch import nn

class CustomizedBCELoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce = nn.BCELoss(reduction="none")
  def forward(self, outputs, labels, attention_mask, device):
    y = self.bce(outputs, labels) # Same shape as input.
    y = torch.where(labels == 1, y, 0.002 * y) # Mask by label.
    zero_cuda = torch.tensor(0, dtype=y.dtype).to(device)
    y = torch.where(attention_mask==0, zero_cuda, y) # Mask pad token loss.
    loss = torch.sum(y)
    return loss