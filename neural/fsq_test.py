import torch
import torch.nn as nn

import math
from vector_quantize_pytorch import FSQ
from lapa import ActionTransformer

if __name__ == '__main__':
    hidden_size = 768
    levels = [8, 6, 5]
    
    linear = nn.Linear(hidden_size, len(levels))
    fan_out = linear.weight.size(0)
    fan_in = linear.weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)
    linear.reset_parameters()
    
    norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
    norm2 = nn.LayerNorm(len(levels))
    
    vq = FSQ(levels=levels)
    
    x = torch.randn(512, hidden_size)
    x = norm(x)
    print(x.mean(), x.std())
    x = linear(x)
    x = norm2(x) * 0.6
    x = x.unsqueeze(1)
    print(x.mean(), x.std())
    x, indices = vq(x)
    
    indices = indices.flatten()
    num_tokens = indices.numel()
    print(indices.shape, num_tokens)

    counts = torch.bincount(indices, minlength=math.prod(levels)).float()
    
    active_mask = counts > 0
    active_count = active_mask.sum().item()
    utilization = active_count / math.prod(levels)

    probs = counts / num_tokens
    print(probs.min(), probs.mean(), probs.std(), probs.max())
    probs = probs + 1e-10
    entropy = -torch.sum(probs * torch.log(probs))
    perplexity = torch.exp(entropy).item()
    
    print(perplexity)
    
    # model = ActionTransformer(16, hidden_size, levels, 48, 2)
    