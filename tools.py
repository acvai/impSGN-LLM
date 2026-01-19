"""
Utility functions for contrastive learning and model optimization.

FUNCTIONS:
1. gen_label(labels)
   Generates a ground truth similarity matrix for contrastive learning.
   For a list of labels, creates a binary matrix where gt[i,k] = 1 
   if labels[i] == labels[k] (same class/cluster).
   
   Args:
       labels (list/array): Class labels for each sample
   Returns:
       gt (numpy.ndarray): Binary similarity matrix of shape (num, num)

2. convert_models_to_fp32(model)
   Converts all model parameters and gradients to 32-bit floating point (FP32).
   Useful for ensuring numerical stability during training.
   
   Args:
       model (torch.nn.Module): PyTorch model to convert

3. convert_models_to_fp16(model)
   Converts all model parameters and gradients to 16-bit floating point (FP16).
   Reduces memory usage and can accelerate computation (mixed precision training).
   
   Args:
       model (torch.nn.Module): PyTorch model to convert

4. create_logits(x1, x2, logit_scale)
   Computes bidirectional cosine similarity logits for contrastive loss.
   Normalizes embeddings, computes cosine similarity, and scales by logit_scale.
   
   Args:
       x1 (torch.Tensor): First set of embeddings
       x2 (torch.Tensor): Second set of embeddings
       logit_scale (torch.Tensor): Learnable temperature parameter
   Returns:
       Tuple[torch.Tensor, torch.Tensor]: (logits_per_x1, logits_per_x2)
       Both tensors have shape [batch_size, batch_size]
       
USAGE NOTES:
- Functions assume PyTorch models/tensors (except gen_label which uses NumPy)
- create_logits() is typically used in CLIP-style contrastive learning
- FP16/FP32 conversion helps manage memory-performance tradeoffs
"""

import numpy

def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2