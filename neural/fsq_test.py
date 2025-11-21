import torch
import torch.nn as nn

import math
from vector_quantize_pytorch import FSQ

if __name__ == '__main__':
    hidden_size = 768
    levels = [8, 8]
    
    linear = nn.Linear(hidden_size, len(levels))
    fan_out = linear.weight.size(0)
    fan_in = linear.weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)
    linear.reset_parameters()
    
    norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
    
    vq = FSQ(levels=levels)
    
    x = torch.randn(64, 32, hidden_size) * 0.02
    x = norm(x)
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
    