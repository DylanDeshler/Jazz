import torch
import torch.nn as nn
import torch.nn.functional as F

from consistency_decoder import ConsistencyDecoderUNet
from seanet import SEANetEncoder
from fm import FM, FMEulerSampler

class DiTo(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_shape = (128, 50)
        self.encoder = SEANetEncoder(n_residual_layers=2, lstm=0, transformer=1)
        self.z_norm = nn.LayerNorm(self.z_shape[0], elementwise_affine=False)
        self.unet = ConsistencyDecoderUNet(in_channels=1, z_dec_channels=128, c0=128, c1=256, c2=512)

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        z = self.encode(x)
        # optional noise synchronization
        loss = self.diffusion.loss(self.unet, x, net_kwargs={'z_dec': z})
        return loss
    
    def encode(self, x):
        z = self.encoder(x)
        z = self.z_norm(z.transpose(1, 2)).transpose(1, 2)
        return z
    
    def sample(self, shape, n_steps=50):
        device = next(self.parameters()).device
        return self.sampler.sample(self.unet, shape, n_steps, net_kwargs={'z_dec': torch.randn(shape[0] + self.z_shape, device=device)})
    
    def reconstruct(self, x, n_steps=50):
        z = self.encode(x)
        return self.sampler.sample(self.unet, x.shape, n_steps, net_kwargs={'z_dec': z})

if __name__ == '__main__':
    device = torch.device('mps')

    model = DiTo().to(device)
    from torchinfo import summary
    summary(model)

    x = torch.randn((16, 1, 16000)).to(device)
    loss = model(x)
    print(loss.item())

    y = model.reconstruct(x)
    print(y.shape)

    y = model.sample(x.shape)
    print(y.shape)