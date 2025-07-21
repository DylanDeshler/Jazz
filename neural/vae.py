# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))

# should be better to invert the transform but qualitatively doesnt seem to be the case...
def instantaneous_frequency_to_phase(x):
    return x
    phase = torch.cumsum(x, dim=-1)
    print(phase.shape)
    # return phase

    group_delay = (phase[:, 1:] - phase[:, :-1]) % 2 * torch.pi
    average_group_delay = torch.mean(group_delay, dim=1, keepdim=True)
    print(average_group_delay.shape)

    phase[:, [0]] = average_group_delay
    return phase

    # inital_phase = 

    # phase = torch.cat([initial_phase, phase], dim=-1)
    # return phase

class STFT(nn.Module):
    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length)
    
    @torch.compile.disable
    def forward(self, x):
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window.to(x.device), return_complex=True, center=True)
        mag, phase = torch.abs(x), torch.angle(x)

        # # global normalization
        mag = (safe_log(mag) + 17) / 23

        phase = torch.from_numpy(np.unwrap(phase.cpu().detach().numpy(), axis=-1)).to(x.device)
        phase = torch.diff(phase, dim=-1, prepend=torch.zeros(*phase.shape[:2], 1).to(x.device))   # temporal differencing

        # drop DC band
        mag = mag[:, 1:, 1:]
        phase = phase[:, 1:, 1:]

        return torch.stack([mag, phase], 1)

class STFTHead(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mel_transform = torchaudio.transforms.MelScale(n_mels=100, sample_rate=16000, n_stft=257)
    
    @torch.compile.disable
    def forward(self, x, targets=None):
        mag, phase = x.chunk(2, 1)

        if targets is not None:
            target_mag, target_phase = targets.chunk(2, 1)
            mag_loss = F.mse_loss(mag, target_mag)  # hinge loss might be best...
            mel_loss = 0#1 * F.mse_loss(self.mel_transform(mag), self.mel_transform(target_mag))

            # # global normalization
            unnorm_mag = torch.exp(target_mag * 23 - 17)
            unnorm_mag = torch.clip(unnorm_mag, max=1e2)

            weight = torch.clamp(torch.abs(unnorm_mag) / (1e-7 + unnorm_mag.flatten(1).max(1)[0].unsqueeze(-1).unsqueeze(-1)), 0.1, 1)
            phase_loss = (1 - torch.cos(phase - target_phase)) * weight
            phase_loss =  0.002 * phase_loss.mean()  # ideal weighting may vary by training iteration 0.001 - 0.01 or 1 at end
            # print(mag_loss.item(), phase_loss.item())
            loss = mag_loss + mel_loss + phase_loss
        else:
            loss = None

        # add DC band as 0
        mag = F.pad(mag, (0, 1, 0, 1))
        phase = F.pad(phase, (0, 1, 0, 1))

        # global normalization
        mag = torch.exp(mag * 23 - 17)
        mag = torch.clip(mag, max=1e2)

        phase = instantaneous_frequency_to_phase(phase)
        x = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return x, loss

class AutoencoderKL(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, in_ch, embed_dim, ch_mult, use_variational=True, ckpt_path=None):
        super().__init__()
        self.in_stft = STFT(n_fft, win_length, hop_length)
        self.out_stft = STFTHead()
        self.encoder = Encoder(in_channels=in_ch, ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder(out_ch=in_ch, ch_mult=ch_mult, z_channels=embed_dim)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.embed_dim = embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["model"]
        msg = self.load_state_dict(sd, strict=False)
        print("Loading pre-trained KL-VAE")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")

    def encode(self, x):
        x = self.in_stft(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec, loss = self.out_stft(dec)
        return dec, loss
    
    def forward(self, x, sample_posterior=True):
        x = self.in_stft(x)
        targets = x.clone()
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec, loss = self.out_stft(dec, targets)
        loss += 1e-6 * posterior.kl()
        return dec, loss
    

import os
from tqdm import tqdm
import requests

def download_pretrained_vae(overwrite=False):
    download_path = "pretrained_models/vae/kl16.ckpt"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/vae", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchinfo import summary
    import librosa

    # download_pretrained_vae()
    checkpoint = torch.load('/Users/dylan.d/Documents/research/music/jazz/neural/pretrained_models/vae/kl16.ckpt')['model']
    model = AutoencoderKL(16, (1, 1, 2, 2, 4), use_variational=True, ckpt_path='/Users/dylan.d/Documents/research/music/jazz/neural/pretrained_models/vae/kl16.ckpt')
    summary(model)

    rate = 16000
    n_fft = 512
    hop_length = int(rate * 10 / 1000)
    win_length = int(rate * 25 / 1000)

    x, sr = librosa.load('/Users/dylan.d/Documents/research/music/jazz_data_16000_full_clean/JV-4422-1928-QmcM2e4W9DtyYbW1GnaD3MgtUioYLCaXooPEB613f89MP5.wav-DV514202.wav')
    x = torch.from_numpy(x[rate*4:rate*6]).unsqueeze(0)

    window = torch.hann_window(win_length)
    x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=True)
    mag, phase = torch.abs(x), torch.angle(x)

    mag = mag.repeat(3, 1, 1).unsqueeze(0)
    # mag = torch.randn(1, 3, 256, 256)
    print(mag.shape)
    z = model.encode(mag)
    out = model.decode(z.sample())
    print(out.shape)

    print(mag.min(), mag.max(), out.min(), out.max())
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(mag.squeeze().permute(1, 2, 0).detach().numpy(), vmin=0, vmax=10)
    ax1.imshow(out.squeeze().permute(1, 2, 0).detach().numpy(), vmin=-1, vmax=10)
    plt.show()