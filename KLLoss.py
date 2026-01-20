"""
KLLoss.py - Stable KL Divergence Loss for impSGN-LLM

Description:
-------------
This module implements a numerically stable KL Divergence loss designed for
contrastive learning between model features and textual embeddings in the
impSGN-LLM framework.

The loss computes:
    KL(log P || Q)  with P = log-softmax(prediction / temperature)
                        Q = softmax(label / temperature)

It uses reduction='batchmean' to ensure proper scaling of the loss over the
batch, avoiding the deprecation warning in PyTorch 2.x and aligning with
standard KL-divergence definitions.

Numerical safeguards are included to prevent NaN/Inf issues during training:
- log probabilities are clamped with `torch.nan_to_num`
- target probabilities are clamped to avoid log(0)
- temperature scaling is supported for softening/ sharpening distributions

Usage:
------
from KLLoss import KLLoss

criterion = KLLoss(temperature=1.0)
loss = criterion(prediction, label)

Changes compared to previous version:
--------------------------------------
1. **Reduction method changed:**
   - OLD: nn.KLDivLoss(size_average=True, reduce=True) → implicitly 'mean'
     ⚠ Triggers warning: "reduction: 'mean' divides total loss by batch and support"
   - NEW: nn.KLDivLoss(reduction='batchmean')
     ✅ Correctly divides by batch only and aligns with KL math

2. **Removed manual multiplication by batch_size**
   - OLD: loss = self.error_metric(...) * batch_size
   - NEW: No manual scaling required; batchmean handles correct normalization

3. **Temperature scaling is now explicit**
   - OLD: label multiplied by 10 arbitrarily
   - NEW: Controlled via `temperature` parameter

4. **Numerical safety**
   - OLD: no safeguards → possible NaN/Inf in loss
   - NEW: uses torch.nan_to_num and torch.clamp for stability

5. **Future-proof**
   - NEW: Compatible with PyTorch 2.x without warnings

This new version stabilizes training compared to the original project GAP, prevents NaNs, and ensures correct KL computation
for contrastive learning in impSGN-LLM.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        print('========= using KL Loss (batchmean, stable) =========')
        self.temperature = temperature
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, prediction, label):
        """
        prediction: logits (B, C)
        label: soft targets (B, C)
        """

        # log-probabilities (student)
        log_p = F.log_softmax(prediction / self.temperature, dim=1)

        # probabilities (teacher / target)
        q = F.softmax(label / self.temperature, dim=1)

        # numerical safety
        log_p = torch.nan_to_num(log_p, nan=0.0, neginf=-1e4, posinf=1e4)
        q = torch.clamp(q, min=1e-8)

        loss = self.kl(log_p, q)
        return loss
