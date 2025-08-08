import torch
import torch.nn as nn
import torch.nn.functional as F

from consistency_decoder import ConsistencyDecoderUNet
from seanet import SEANetEncoder, SEANetDecoder
from fm import FM, FMEulerSampler

class DiTo(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SEANetEncoder(n_residual_layers=2, lstm=0, transformer=1)
        # self.decoder = SEANetDecoder(n_residual_layers=2, lstm=0, transformer=1)
        self.z_norm = nn.LayerNorm(128, elementwise_affine=False)

        self.unet = ConsistencyDecoderUNet(in_channels=1, z_dec_channels=128, c0=128, c1=256, c2=512)
    
    def forward(self, x, t):
        z = self.encoder(x)
        z = self.z_norm(z.transpose(1, 2)).transpose(1, 2)
        out = self.unet(x=x, t=t, z_dec=z)
        return out
    
    def reconstruct(self, x, x_t, t):
        z = self.encoder(x)
        z = self.z_norm(z.transpose(1, 2)).transpose(1, 2)
        out = self.unet(x_t, t=t, z_dec=z)
        return out

class DiToTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiTo()
        self.diffusion = FM()
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        loss, x_t, y = self.diffusion.loss(self.model, x, return_all=True)
        return y, loss

    def sample(self, shape, n_steps=50):
        return self.sampler.sample(self.model, shape, n_steps)
    
    def reconstruct(self, x, n_steps=50):
        return self.sampler.reconstruct(self.model, x, n_steps)

if __name__ == '__main__':
    device = torch.device('mps')

    model = DiToTrainer().to(device)
    from torchinfo import summary
    summary(model)

    x = torch.randn((16, 1, 16000)).to(device)
    out = model(x)
    print(out.shape)

    y = model.reconstruct(x)
    print(y.shape)