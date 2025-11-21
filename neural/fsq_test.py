import torch
import torch.nn as nn

import math
from vector_quantize_pytorch import FSQ

if __name__ == '__main__':
    hidden_size = 768
    levels = [8, 8]
    
    linear = nn.Linear(hidden_size, len(levels))
    vq = FSQ(levels=levels)
    
    x = torch.randn(64, 32, hidden_size)
    x = linear(x)
    x, indices = vq(x)
    
    indices = indices.flatten()
    num_tokens = indices.numel()

    counts = torch.bincount(indices, minlength=math.prod(levels)).float()
    
    active_mask = counts > 0
    active_count = active_mask.sum().item()
    utilization = active_count / math.prod(levels)

    probs = counts / num_tokens
    probs = probs + 1e-10
    entropy = -torch.sum(probs * torch.log(probs))
    perplexity = torch.exp(entropy).item()
    
    print(perplexity)
    