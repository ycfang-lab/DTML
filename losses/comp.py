import torch
from torch import nn
import torch.nn.functional as F


class CompLoss(nn.Module):
    def __init__(self, temperature=1.0, class_mode=False):
        super(CompLoss, self).__init__()
        self.temperature = temperature
        self.class_mode = class_mode
    
    @staticmethod
    def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
 
    def forward(self, x1, x2) -> torch.Tensor:
        # normalized features
        x1 = x1 / x1.norm(p=2, dim=-1, keepdim=True)
        x2 = x2 / x2.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity
        if self.class_mode:
            similarity = torch.matmul(x1.t(), x2) * self.temperature
        else:
            similarity = torch.matmul(x1, x2.t()) * self.temperature
        x1_loss = self.contrastive_loss(similarity)
        x2_loss = self.contrastive_loss(similarity.t())

        return (x1_loss + x2_loss) / 2.0
