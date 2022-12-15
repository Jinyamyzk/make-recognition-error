import torch 
from torch import nn

class CustomizedBCELoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce = nn.BCELoss(reduction="none")
  def forward(self, outputs, labels):
    y = self.bce(outputs, labels) # Same shape as input.
    y = torch.where(labels == 1, y, 0.002 * y) # Mask by label.
    loss = torch.sum(y)
    return loss