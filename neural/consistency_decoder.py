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
            Downsample(channels[0], t_dim, ratios[0]),
        ]))

        # Levels 1..N-1
        for i in range(1, len(channels)):
            c_prev = channels[i - 1]
            c_cur = channels[i]
            self.down.append(nn.ModuleList([
                ConvResblock(c_prev, c_cur, t_dim),
                ConvResblock(c_cur, c_cur, t_dim),
                ConvResblock(c_cur, c_cur, t_dim),
                Downsample(c_cur, t_dim, ratios[i]),
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

        # up_3 = nn.ModuleList([
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     Upsample(c2, t_dim, ratios[2]),
        # ])
        # up_2 = nn.ModuleList([
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 * 2, c2, t_dim),
        #     ConvResblock(c2 + c1, c2, t_dim),
        #     Upsample(c2, t_dim, ratios[1]),
        # ])
        # up_1 = nn.ModuleList([
        #     ConvResblock(c2 + c1, c1, t_dim),
        #     ConvResblock(c1 * 2, c1, t_dim),
        #     ConvResblock(c1 * 2, c1, t_dim),
        #     ConvResblock(c0 + c1, c1, t_dim),
        #     Upsample(c1, t_dim, ratios[0]),
        # ])
        # up_0 = nn.ModuleList([
        #     ConvResblock(c0 + c1, c0, t_dim),
        #     ConvResblock(c0 * 2, c0, t_dim),
        #     ConvResblock(c0 * 2, c0, t_dim),
        #     ConvResblock(c0 * 2, c0, t_dim),
        # ])

        self.up = nn.ModuleList([])
        self.up.append(nn.ModuleList([
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            Upsample(channels[-1], t_dim, ratios[-1]),
        ]))
        self.up.append(nn.ModuleList([
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] * 2, channels[-1], t_dim),
            ConvResblock(channels[-1] + channels[-2], channels[-1], t_dim),
            Upsample(channels[-1], t_dim, ratios[-2]),
        ]))

        for i in range(1, len(channels) - 1):
            c_prev = channels[-i]
            c_cur = channels[-i-1]
            c_next = channels[-i-2];print(i, len(channels), c_prev, c_cur, c_next)
            self.up.append(nn.ModuleList([
                ConvResblock(c_prev + c_cur, c_cur, t_dim),
                ConvResblock(c_cur * 2, c_cur, t_dim),
                ConvResblock(c_cur * 2, c_cur, t_dim),
                ConvResblock(c_next + c_cur, c_cur, t_dim),
                Upsample(c_cur, t_dim, ratios[-i-2]),
            ]))
        
        self.up.append(nn.ModuleList([
            ConvResblock(channels[0] + channels[1], channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
            ConvResblock(channels[0] * 2, channels[0], t_dim),
        ]))

        # for i in range(n_levels):
        #     j = -(i + 1)
        #     if i == 0:
        #         self.up.append(nn.ModuleList([
        #             ConvResblock(channels[j] * 2, channels[j], t_dim),
        #             ConvResblock(channels[j] * 2, channels[j], t_dim),
        #             ConvResblock(channels[j] * 2, channels[j], t_dim),
        #             ConvResblock(channels[j] * 2, channels[j], t_dim),
        #             Upsample(channels[j], t_dim, ratios[j]),
        #         ]))
        #     elif i == n_levels - 1:
        #          self.up.append(nn.ModuleList([
        #             ConvResblock(channels[0] + channels[1], channels[0], t_dim),
        #             ConvResblock(channels[0] * 2, channels[0], t_dim),
        #             ConvResblock(channels[0] * 2, channels[0], t_dim),
        #             ConvResblock(channels[0] * 2, channels[0], t_dim),
        #         ]))
        #     else:
        #         self.up.append(nn.ModuleList([
        #             ConvResblock(c2 + c1, c1, t_dim),
        #             ConvResblock(c1 * 2, c1, t_dim),
        #             ConvResblock(c1 * 2, c1, t_dim),
        #             ConvResblock(channels[0] + c1, c1, t_dim),
        #             Upsample(c1, t_dim, ratios[0]),
        #         ]))

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
                print('down: ', x.shape)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)
            print('mid: ', x.shape)

        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, ConvResblock):
                    print('up: ', x.shape, skips[-1].shape)
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)
                print('up: ', x.shape)

        return self.output(x)