import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class EMATracker:
    """Tracks the exponential moving average of gradient norms."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.emas = {}

    def update(self, task_name, current_norm):
        if task_name not in self.emas:
            self.emas[task_name] = current_norm
        else:
            self.emas[task_name] = self.beta * self.emas[task_name] + (1 - self.beta) * current_norm
        return self.emas[task_name]

class GradientBalancer(torch.autograd.Function):
    """
    Intercepts the backward pass to scale gradients based on their EMA norm.
    """
    @staticmethod
    def forward(ctx, x, ema_tracker, task_name, task_weight):
        # Store objects for the backward pass
        ctx.ema_tracker = ema_tracker
        ctx.task_name = task_name
        ctx.task_weight = task_weight
        
        # The forward pass does absolutely nothing to the tensor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Calculate the L2 norm of the incoming gradient from the head
        norm = torch.linalg.norm(grad_output)
        
        # 2. Update the EMA for this specific task
        ema_norm = ctx.ema_tracker.update(ctx.task_name, norm)
        
        # 3. Scale the gradient: (Weight / EMA) * Gradient
        # We add 1e-8 to prevent division by zero
        scale_factor = ctx.task_weight / (ema_norm + 1e-8)
        scaled_grad = grad_output * scale_factor
        
        # Return scaled grad for 'x', and None for the other non-tensor arguments
        return scaled_grad, None, None, None

class DropPath(nn.Module):
    """Stochastic Depth: Randomly drops paths (blocks) per sample during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (Batch, 1, 1) to broadcast across channels and time
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
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
        self.mu = -34.36543
        self.std = 15.82586
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.transform(x)
        x = (x - self.mu) / (self.std + 1e-6)
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted for 2D Audio Spectrograms"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        
        return input + x

class MultiTaskFAD(nn.Module):
    def __init__(self, num_instruments, num_labels, bpm_bins, year_bins, n_fft=1024, hop_length=512, n_mels=192, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        
        self.to_mel = ToMel(16000, n_fft, hop_length, n_mels)
        self.ema_tracker = EMATracker(beta=0.99)
        
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
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self.head_bpm = nn.Linear(dims[-1], bpm_bins)
        self.head_year = nn.Linear(dims[-1], year_bins)
        self.head_instruments = nn.Linear(dims[-1], num_instruments) 
        self.head_label = nn.Linear(dims[-1], num_labels)

    def forward_features(self, x):
        x = self.to_mel(x)
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

    def forward(self, x, targets, task_weights):
        features = self.forward_features(x)
        
        outputs = {}
        bpm = GradientBalancer.apply(features, self.ema_tracker, 'bpm', task_weights['bpm'])
        bpm = self.head_bpm(bpm)
        outputs['bpm'] = bpm
        
        year = GradientBalancer.apply(features, self.ema_tracker, 'year', task_weights['year'])
        year = self.head_year(year)
        outputs['year'] = year
        
        inst = GradientBalancer.apply(features, self.ema_tracker, 'instruments', task_weights['instruments'])
        inst = self.head_instruments(inst)
        outputs['instruments'] = inst
        
        label = GradientBalancer.apply(features, self.ema_tracker, 'label', task_weights['label'])
        label = self.head_label(label)
        outputs['label'] = label
        
        loss_bpm = F.kl_div(F.log_softmax(bpm, dim=-1), targets['bpm'], reduction='batchmean')
        loss_year = F.kl_div(F.log_softmax(year, dim=-1), targets['year'], reduction='batchmean')
        loss_inst = F.binary_cross_entropy_with_logits(inst, targets['inst'])
        loss_label = F.cross_entropy(label, targets['label'])

        total_loss = loss_bpm + loss_year + loss_inst + loss_label

        outputs['loss'] = {
            'total': total_loss,
            'bpm': loss_bpm,
            'year': loss_year,
            'inst': loss_inst,
            'label': loss_label
        }
        
        return outputs