import torch
import torch.nn as nn
import torch.nn.functional as F

from consistency_decoder import ConsistencyDecoderUNet
from seanet import SEANetEncoder
from fm import FM, FMEulerSampler

class DiTo(nn.Module):
    def __init__(self, z_shape, n_residual_layers, lstm, transformer, channels = 1, dimension = 128, n_filters = 32, ratios = [8, 5, 4, 2], c0=128, c1=256, c2=512):
        super().__init__()
        self.z_shape = z_shape
        self.encoder = SEANetEncoder(channels=channels, dimension=dimension, n_filters=n_filters, ratios=ratios, n_residual_layers=n_residual_layers, lstm=lstm, transformer=transformer)
        self.z_norm = nn.LayerNorm(self.z_shape[0], elementwise_affine=False)
        self.unet = ConsistencyDecoderUNet(in_channels=channels, z_dec_channels=dimension, c0=c0, c1=c1, c2=c2, ratios=ratios[:3])

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        z = self.encode(x)
        
        # noise synchronization
        t = torch.rand(z.shape[0], device=z.device)
        post_t = t + torch.rand(z.shape[0], device=z.device) * (1 - t)
        z_t, _ = self.diffusion.add_noise(z, t)
        mask_aug = (torch.rand(z.shape[0], device=z.device) < 0.1).float()
        z = mask_aug.view(-1, 1, 1) * z_t + (1 - mask_aug).view(-1, 1, 1) * z
        t[mask_aug.long()] = post_t
        
        loss = self.diffusion.loss(self.unet, x, t, net_kwargs={'z_dec': z})
        return loss
    
    def encode(self, x):
        z = self.encoder(x)
        z = self.z_norm(z.transpose(1, 2)).transpose(1, 2)
        return z

    def reconstruct(self, x, n_steps=50):
        z = self.encode(x)
        return self.decode(z, shape=x.shape, n_steps=n_steps)
    
    def decode(self, z, shape=(1, 16000), n_steps=50):
        return self.sampler.sample(self.unet, (z.shape[0], *shape[-2:]), n_steps, net_kwargs={'z_dec': z})

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