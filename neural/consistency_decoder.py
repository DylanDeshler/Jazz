# https://gist.github.com/mrsteyk/74ad3ec2f6f823111ae4c90e168505ac

import torch
import torch.nn.functional as F
import torch.nn as nn



class TimestepEmbedding(nn.Module):
    def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_time, n_emb)
        self.f_1 = nn.Linear(n_emb, n_out)
        self.f_2 = nn.Linear(n_out, n_out)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, pe_dim=320, out_dim=1280, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = pe_dim
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.f_1 = nn.Linear(pe_dim, out_dim)
        self.f_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=3) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
    def __init__(self, in_features, out_features, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv1d(out_features, out_features, kernel_size=3, padding=1)

        skip_conv = in_features != out_features
        self.f_s = (
            nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
            if skip_conv
            else nn.Identity()
        )

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2) + 1
        t_2 = t[1].unsqueeze(dim=2)

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class Downsample(nn.Module):
    def __init__(self, in_channels, t_dim, ratio=2) -> None:
        super().__init__()
        self.ratio = ratio
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool1d = F.avg_pool1d(gn_1, kernel_size=self.ratio, stride=None)
        f_1 = self.f_1(avg_pool1d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool1d(x_skip, kernel_size=self.ratio, stride=None)


# Also ConvResblock
class Upsample(nn.Module):
    def __init__(self, in_channels, t_dim, ratio=2) -> None:
        super().__init__()
        self.ratio = ratio
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        upsample = F.interpolate(gn_1, scale_factor=self.ratio, mode='nearest')
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)
        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.interpolate(x_skip.float(), scale_factor=self.ratio, mode='nearest').to(x_skip.dtype)


class ConsistencyDecoderUNet(nn.Module):
    def __init__(self, in_channels=3, z_dec_channels=None, c0=320, c1=640, c2=1024, pe_dim=320, t_dim=1280, ratios=[8, 5, 4]) -> None:
        super().__init__()
        if z_dec_channels is not None:
            in_channels += z_dec_channels
        self.embed_image = ImageEmbedding(in_channels=in_channels, out_channels=c0)
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)

        down_0 = nn.ModuleList([
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            Downsample(c0, t_dim, ratios[0]),
        ])
        down_1 = nn.ModuleList([
            ConvResblock(c0, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            Downsample(c1, t_dim, ratios[1]),
        ])
        down_2 = nn.ModuleList([
            ConvResblock(c1, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            Downsample(c2, t_dim, ratios[2]),
        ])
        down_3 = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])
        self.down = nn.ModuleList([
            down_0,
            down_1,
            down_2,
            down_3,
        ])

        self.mid = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])

        up_3 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            Upsample(c2, t_dim, ratios[2]),
        ])
        up_2 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 + c1, c2, t_dim),
            Upsample(c2, t_dim, ratios[1]),
        ])
        up_1 = nn.ModuleList([
            ConvResblock(c2 + c1, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c0 + c1, c1, t_dim),
            Upsample(c1, t_dim, ratios[0]),
        ])
        up_0 = nn.ModuleList([
            ConvResblock(c0 + c1, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
        ])
        self.up = nn.ModuleList([
            up_0,
            up_1,
            up_2,
            up_3,
        ])

        self.output = ImageUnembedding(in_channels=c0, out_channels=1)
    
    def get_last_layer_weight(self):
        return self.output.f.weight
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output.gn.weight, 0)
        nn.init.constant_(self.output.gn.bias, 0)
        nn.init.constant_(self.output.f.weight, 0)
        nn.init.constant_(self.output.f.bias, 0)

    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        if z_dec is not None:
            # if z_dec.shape[-2] != x.shape[-2] or z_dec.shape[-1] != x.shape[-1]:
            if z_dec.shape[-1] != x.shape[-1]:
                # assert x.shape[-2] // z_dec.shape[-2] == x.shape[-1] // z_dec.shape[-1]
                # z_dec = F.upsample_nearest(z_dec, scale_factor=x.shape[-2] // z_dec.shape[-2])
                z_dec = F.interpolate(z_dec, scale_factor=x.shape[-1] // z_dec.shape[-1], mode='nearest')
            x = torch.cat([x, z_dec], dim=1)
        
        x = self.embed_image(x)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)

        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor
        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x

class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int = 3
    ):
        super().__init__()
        assert out_channels % factor == 0, f'{out_channels}, {factor}'
        # self.conv = ConvLayer(
        #     in_channels=in_channels,
        #     out_channels=out_channels // out_ratio,
        #     kernel_size=kernel_size,
        #     use_bias=True,
        #     norm=None,
        #     act_func=None,
        # )
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels // factor, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_unshuffle = PixelUnshuffle1D(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        x = self.pixel_unshuffle(x)
        return x

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert in_channels * factor % out_channels == 0, f'{in_channels} {factor} {out_channels}'
        self.group_size = in_channels * factor // out_channels
        self.pixel_unshuffle = PixelUnshuffle1D(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_unshuffle(x)
        B, C, L = x.shape
        x = x.view(B, self.out_channels, self.group_size, L)
        x = x.mean(dim=2)
        return x

class DownsampleV3(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super().__init__()
        self.conv = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, ratio, ratio * 2)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, ratio)
    
    def forward(self, x):
        print(x.shape)
        print(self.conv(x).shape)
        print(self.shortcut(x).shape)
        x = self.conv(x) + self.shortcut(x)
        return x

class DownsampleV2(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, ratio=2) -> None:
        super().__init__()
        self.ratio = ratio

        self.conv = ConvPixelUnshuffleDownSampleLayer(in_channels, in_channels, ratio)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, in_channels, ratio)

        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool1d = self.conv(gn_1);print(gn_1.shape, avg_pool1d.shape)
        f_1 = self.f_1(avg_pool1d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + self.shortcut(x_skip)

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int = 3
    ):
        super().__init__()
        # self.conv = ConvLayer(
        #     in_channels=in_channels,
        #     out_channels=out_channels * out_ratio,
        #     kernel_size=kernel_size,
        #     use_bias=True,
        #     norm=None,
        #     act_func=None,
        # )
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels * factor, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = PixelShuffle1D(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels * factor % in_channels == 0, f'{out_channels} {factor} {in_channels}'
        self.repeats = out_channels * factor // in_channels
        self.pixel_shuffle = PixelShuffle1D(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = self.pixel_shuffle(x)
        return x

class UpsampleV3(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super().__init__()
        self.conv = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, ratio, ratio * 2)
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, ratio)
    
    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x

class UpsampleV2(nn.Module):
    def __init__(self, in_channels, t_dim, ratio=2) -> None:
        super().__init__()
        self.ratio = ratio
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.conv = ConvPixelShuffleUpSampleLayer(in_channels, in_channels, ratio)
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, in_channels, ratio)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        upsample = self.conv(gn_1)
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)
        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + self.shortcut(x_skip)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNConvBlock(nn.Module):
    def __init__(self, hidden_features, t_dim) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, hidden_features)
        self.conv1 = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 3 * hidden_features, bias=True)
        )

    def forward(self, x, t):
        x_skip = x
        gate, shift, scale = self.adaLN_modulation(t).chunk(3, dim=-1)

        x = modulate(self.norm(x), shift, scale)
        x = F.silu(self.conv1(x))
        x = gate.unsqueeze(1) * self.conv2(x)

        return x_skip + x

class DylanDecoderUNet(nn.Module):
    def __init__(self, in_channels=3, z_dec_channels=None, channels=[320, 640, 1024], pe_dim=320, t_dim=1280, ratios=[8, 5, 4]) -> None:
        super().__init__()
        if z_dec_channels is not None:
            in_channels += z_dec_channels
        self.embed_image = ImageEmbedding(in_channels=in_channels, out_channels=channels[0])
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)

        assert len(channels) == len(ratios), f'{len(channels)} != {len(ratios)}'
        depths = [3] * len(channels)

        self.down = nn.ModuleList([])
        for i, (channel, depth, ratio) in enumerate(zip(channels, depths, ratios)):
            blocks = nn.ModuleList([])
            for _ in range(depth):
                blocks.append(AdaLNConvBlock(channel, t_dim))
            if i < len(channels) - 1:
                blocks.append(DownsampleV3(channel, channels[i + 1], ratio))
            self.down.append(blocks)
        
        self.mid = nn.ModuleList([
            AdaLNConvBlock(channels[-1], t_dim) for _ in range(depths[-1])
        ])

        depths = [4] * len(channels)
        self.up = nn.ModuleList([])
        for i, (channel, depth, ratio) in enumerate(zip(channels, depths, ratios)):
            blocks = nn.ModuleList([])
            for _ in range(depth):
                blocks.append(AdaLNConvBlock(channel, t_dim))
            if i < len(channels) - 1:
                blocks.append(UpsampleV3(channel, channels[i + 1], ratio))
            self.up.append(blocks)
        self.up = self.up[::-1]

        self.output = ImageUnembedding(in_channels=channels[0], out_channels=1)
    
    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        if z_dec is not None:
            if z_dec.shape[-1] != x.shape[-1]:
                z_dec = F.interpolate(z_dec, scale_factor=x.shape[-1] // z_dec.shape[-1], mode='nearest')
            x = torch.cat([x, z_dec], dim=1)
        
        x = self.embed_image(x)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)

        for up in self.up:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)

if __name__ == '__main__':
    model = DylanDecoderUNet(in_channels=1, z_dec_channels=None, channels=[128, 192, 256, 512, 1024], ratios=[8, 4, 4, 4, 2, 2])

class ConsistencyDecoderUNetV2(nn.Module):
    def __init__(self, in_channels=3, z_dec_channels=None, channels=[320, 640, 1024], pe_dim=320, t_dim=1280, ratios=[8, 5, 4]) -> None:
        super().__init__()
        if z_dec_channels is not None:
            in_channels += z_dec_channels
        self.embed_image = ImageEmbedding(in_channels=in_channels, out_channels=channels[0])
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)

        assert len(channels) == len(ratios), f'{len(channels)} != {len(ratios)}'

        self.down = nn.ModuleList([])
        self.down.append(nn.ModuleList([
            ConvResblock(channels[0], channels[0], t_dim),
            ConvResblock(channels[0], channels[0], t_dim),
            ConvResblock(channels[0], channels[0], t_dim),
            DownsampleV2(channels[0], t_dim, ratios[0]),
        ]))

        # Levels 1..N-1
        for i in range(1, len(channels)):
            c_prev = channels[i - 1]
            c_cur = channels[i]
            self.down.append(nn.ModuleList([
                ConvResblock(c_prev, c_cur, t_dim),
                ConvResblock(c_cur, c_cur, t_dim),
                ConvResblock(c_cur, c_cur, t_dim),
                DownsampleV2(c_cur, t_dim, ratios[i]),
            ]))

        # Bottom (no downsample), uses last channel again
        c_bot = channels[-1]
        self.down.append(nn.ModuleList([
            ConvResblock(c_bot, c_bot, t_dim),
            ConvResblock(c_bot, c_bot, t_dim),
            ConvResblock(c_bot, c_bot, t_dim),
        ]))
        
        self.mid = nn.ModuleList([
            ConvResblock(channels[-1], channels[-1], t_dim),
            ConvResblock(channels[-1], channels[-1], t_dim),
        ])

        self.up = nn.ModuleList([])
        self.up.append(nn.ModuleList([
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            UpsampleV2(channels[-1], t_dim, ratios[-1]),
        ]))
        self.up.append(nn.ModuleList([
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] + channels[-2], channels[-1], t_dim),
            UpsampleV2(channels[-1], t_dim, ratios[-2]),
        ]))

        for i in range(1, len(channels) - 1):
            c_prev = channels[-i]
            c_cur = channels[-i-1]
            c_next = channels[-i-2]
            self.up.append(nn.ModuleList([
                ConvResblock(c_prev + c_cur, c_cur, t_dim),
                ConvResblock(c_cur * 2, c_cur, t_dim),
                ConvResblock(c_cur * 2, c_cur, t_dim),
                ConvResblock(c_next + c_cur, c_cur, t_dim),
                UpsampleV2(c_cur, t_dim, ratios[-i-2]),
            ]))
        
        self.up.append(nn.ModuleList([
            ConvResblock(channels[0] + channels[1], channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
        ]))

        self.output = ImageUnembedding(in_channels=channels[0], out_channels=1)
    
    def get_last_layer_weight(self):
        return self.output.f.weight

    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        if z_dec is not None:
            # if z_dec.shape[-2] != x.shape[-2] or z_dec.shape[-1] != x.shape[-1]:
            if z_dec.shape[-1] != x.shape[-1]:
                # assert x.shape[-2] // z_dec.shape[-2] == x.shape[-1] // z_dec.shape[-1]
                # z_dec = F.upsample_nearest(z_dec, scale_factor=x.shape[-2] // z_dec.shape[-2])
                z_dec = F.interpolate(z_dec, scale_factor=x.shape[-1] // z_dec.shape[-1], mode='nearest')
            x = torch.cat([x, z_dec], dim=1)
        
        x = self.embed_image(x)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)

        for up in self.up:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)