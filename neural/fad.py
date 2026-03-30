import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

import torch
import torch.nn as nn

class _GradientBalancerFunction(torch.autograd.Function):
    """
    The hidden autograd engine that intercepts the gradient flowing from 
    a specific head down into the shared trunk, scaling it on the fly.
    """
    @staticmethod
    def forward(ctx, features, total_buffer, fix_buffer, task_weight, total_weight, ema_decay, total_norm, epsilon):
        # Save variables for the backward pass
        ctx.total_buffer = total_buffer
        ctx.fix_buffer = fix_buffer
        ctx.task_weight = task_weight
        ctx.total_weight = total_weight
        ctx.ema_decay = ema_decay
        ctx.total_norm = total_norm
        ctx.epsilon = epsilon
        
        # Pass features through untouched during the forward pass
        return features.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Compute per-batch-item norm (EnCodec style)
        dims = tuple(range(1, grad_output.dim()))
        norm = grad_output.norm(dim=dims).mean()
        batch_size = grad_output.shape[0]
        
        # 2. Update EMA buffers IN-PLACE 
        # (Using pure tensor ops: no float(), no .item() -> zero graph breaks!)
        ctx.total_buffer.mul_(ctx.ema_decay).add_(norm * batch_size)
        ctx.fix_buffer.mul_(ctx.ema_decay).add_(batch_size)
        
        # 3. Calculate the smoothed average norm
        avg_norm = ctx.total_buffer / ctx.fix_buffer
        
        # 4. Calculate the EnCodec scaling factor
        ratio = ctx.task_weight / ctx.total_weight
        scale = ratio * ctx.total_norm / (ctx.epsilon + avg_norm)
        
        # 5. Scale the gradient before it passes down into the trunk
        grad_input = grad_output * scale
        
        # Return gradients for the inputs (None for the hyperparameter arguments)
        return grad_input, None, None, None, None, None, None, None


class GradientBalancer(nn.Module):
    """
    A drop-in module that wraps the EnCodec balancing math into an automatic layer.
    """
    def __init__(self, weights: dict, ema_decay=0.999, total_norm=1.0, epsilon=1e-12):
        super().__init__()
        self.weights = weights
        self.ema_decay = ema_decay
        self.total_norm = total_norm
        self.epsilon = epsilon
        self.total_weight = sum(weights.values())
        
        # Register EMA trackers as PyTorch buffers so they live on the GPU 
        # and are saved in your model's state_dict
        for task in weights.keys():
            self.register_buffer(f'total_{task}', torch.tensor(0.0))
            self.register_buffer(f'fix_{task}', torch.tensor(0.0))

    def forward(self, features, task_name):
        """
        Pass the trunk features through this function before sending them to a head.
        """
        # Fetch the specific EMA buffers for this task
        total_buffer = getattr(self, f'total_{task_name}')
        fix_buffer = getattr(self, f'fix_{task_name}')
        task_weight = self.weights[task_name]
        
        return _GradientBalancerFunction.apply(
            features, 
            total_buffer, 
            fix_buffer, 
            task_weight, 
            self.total_weight, 
            self.ema_decay, 
            self.total_norm, 
            self.epsilon
        )

class DropPath(nn.Module):
    """Stochastic Depth: Randomly drops paths (blocks) per sample during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ToMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__()
        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=40.0,
                f_max=sample_rate // 2,
                power=2.0,
                normalized=True,      # Normalizes the STFT to be magnitude invariant
                center=True,          # Padding to keep time/length consistent
                pad_mode='reflect'    # Better for audio boundary artifacts
            ),
            T.AmplitudeToDB(top_db=80.0)
        )
        # self.mu = -34.36543
        # self.std = 15.82586
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.transform(x)
        
        mu = x.mean((-1, -2), keepdims=True)
        std = x.std((-1, -2), keepdims=True)
        x = (x - mu) / (std + 1e-6)
        # x = (x - self.mu) / (self.std + 1e-6)
        return x

class SpecAugment(nn.Module):
    def __init__(self, time_length=32, frequency_length=64):
        super().__init__()
        self.time_mask = T.TimeMasking(time_length)
        self.freq_mask = T.FrequencyMasking(frequency_length)
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.time_mask(x)
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        return input + self.drop_path(x)

class MultiTaskFAD(nn.Module):
    def __init__(self, num_instruments, num_labels, bpm_bins, year_bins, n_fft=1024, hop_length=512, n_mels=192, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        
        self.to_mel = ToMel(16000, n_fft, hop_length, n_mels)
        self.augment = SpecAugment()
        
        weights = {'bpm': 1, 'label': 1, 'year': 1, 'inst': 1}
        self.balancer = GradientBalancer(weights=weights)
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self.head_bpm = nn.Linear(dims[-1], bpm_bins)
        self.head_year = nn.Linear(dims[-1], year_bins)
        self.head_instruments = nn.Linear(dims[-1], num_instruments) 
        self.head_label = nn.Linear(dims[-1], num_labels)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.to_mel(x)
        
        if self.training:
            x = self.augment(x)
        
        for i in range(4):
            if i == 0:
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 2, 3, 1)
                x = self.downsample_layers[i][1](x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(0, 2, 3, 1)
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 3, 1, 2)
                x = self.downsample_layers[i][1](x)
                
            x = self.stages[i](x)
            
        x = x.mean([-2, -1])
        x = self.norm(x)
        
        return x

    def forward(self, x, targets, target_masks):
        features = self.forward_features(x)
        
        outputs = {}
        bpm = self.head_bpm(self.balancer(features, 'bpm'))
        outputs['bpm'] = bpm
        
        year = self.head_year(self.balancer(features, 'year'))
        outputs['year'] = year
        
        inst = self.head_instruments(self.balancer(features, 'inst'))
        outputs['instruments'] = inst
        
        label = self.head_label(self.balancer(features, 'label'))
        outputs['label'] = label
        
        loss_bpm = F.kl_div(F.log_softmax(bpm, dim=-1), targets['bpm'], reduction='none')
        loss_bpm = loss_bpm.sum(dim=-1) * target_masks['bpm']
        if target_masks['bpm'].sum() > 0:
            loss_bpm = loss_bpm.sum() / target_masks['bpm'].sum()
        else:
            loss_bpm = torch.tensor(0.0, device=features.device, requires_grad=True)
            
        loss_year = F.kl_div(F.log_softmax(year, dim=-1), targets['year'], reduction='none')
        loss_year = loss_year.sum(dim=-1) * target_masks['year']
        if target_masks['year'].sum() > 0:
            loss_year = loss_year.sum() / target_masks['year'].sum()
        else:
            loss_year = torch.tensor(0.0, device=features.device, requires_grad=True)
        
        alpha = 0.4
        smooth_targets = targets['inst'] * (1.0 - alpha) + (alpha / 2.0)
        loss_inst = F.binary_cross_entropy_with_logits(inst, smooth_targets)
        
        loss_label = F.cross_entropy(label, targets['label'])

        total_loss = loss_inst + loss_label + loss_bpm + loss_year

        outputs['loss'] = {
            'total': total_loss,
            'bpm': loss_bpm,
            'year': loss_year,
            'inst': loss_inst,
            'label': loss_label
        }
        
        return outputs