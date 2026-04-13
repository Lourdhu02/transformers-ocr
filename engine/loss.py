import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCTCLoss(nn.Module):
    def __init__(self, blank: int = 0, gamma: float = 2.0,
                 label_smoothing: float = 0.1, zero_infinity: bool = True):
        super().__init__()
        self.blank           = blank
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.ctc             = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity,
                                          reduction="none")

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        if self.label_smoothing > 0:
            T, B, C      = log_probs.shape
            smooth_loss  = -log_probs.mean(dim=-1).mean(dim=0)
            loss         = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        pt      = torch.exp(-loss.detach())
        focal_w = (1 - pt) ** self.gamma
        return (focal_w * loss).mean()
