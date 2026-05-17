import math
import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim):
    """Sinusoidal time embedding, same construction as the Transformer paper."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        x = sinusoidal_embedding(t, self.dim)
        return self.proj(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return self.downsample(x), x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.attn(x)


class UNet(nn.Module):
    """
    Noise-prediction UNet for 32x32 RGB images.
    Channels at each resolution: [64, 128, 256, 256].
    """
    def __init__(self, base_ch=64, time_dim=256):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        td = time_dim * 4

        self.time_emb = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv2d(3, ch[0], 3, padding=1)

        # encoder: 32->16->8->4
        self.down1 = Down(ch[0], ch[1], td, use_attn=False)   # 32->16
        self.down2 = Down(ch[1], ch[2], td, use_attn=True)    # 16->8
        self.down3 = Down(ch[2], ch[3], td, use_attn=True)    # 8->4

        # bottleneck
        self.mid_res1 = ResBlock(ch[3], ch[3], td)
        self.mid_attn = SelfAttention(ch[3])
        self.mid_res2 = ResBlock(ch[3], ch[3], td)

        # decoder
        self.up3 = Up(ch[3], ch[3], ch[2], td, use_attn=True)  # 4->8
        self.up2 = Up(ch[2], ch[2], ch[1], td, use_attn=True)  # 8->16
        self.up1 = Up(ch[1], ch[1], ch[0], td, use_attn=False) # 16->32

        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv2d(ch[0], 3, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        x = self.input_conv(x)

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        return self.out_conv(nn.functional.silu(self.out_norm(x)))
