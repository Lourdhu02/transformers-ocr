import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {
    "nano":  dict(dims=(32,  64,  128), depths=(2, 3, 4),  num_heads=(1, 2, 4),  drop=0.05),
    "tiny":  dict(dims=(64,  128, 256), depths=(3, 6, 3),  num_heads=(2, 4, 8),  drop=0.05),
    "small": dict(dims=(96,  192, 384), depths=(3, 6, 6),  num_heads=(3, 6, 12), drop=0.08),
    "base":  dict(dims=(128, 256, 512), depths=(3, 6, 9),  num_heads=(4, 8, 16), drop=0.10),
}


class MixingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, local_k=7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_k,
                                    padding=local_k // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        global_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        local_out  = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(global_out + local_out)


class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = MixingAttention(dim, num_heads)
        self.norm2  = nn.LayerNorm(dim)
        mlp_dim     = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureRearrangement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)
        x = self.proj(x)
        return x


class SemanticGuidance(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x).log_softmax(2)


class SVTRNet(nn.Module):
    def __init__(self, img_h, img_w, num_classes,
                 dims=(64, 128, 256), depths=(3, 6, 3),
                 num_heads=(2, 4, 8), drop=0.1,
                 use_frm=True, use_sgm=True):
        super().__init__()
        self.use_frm = use_frm
        self.use_sgm = use_sgm

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1), nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            *[SVTRBlock(dims[0], num_heads[0], drop=drop) for _ in range(depths[0])]
        )
        self.merge1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, (2, 1), 1), nn.GELU()
        )

        self.stage2 = nn.Sequential(
            *[SVTRBlock(dims[1], num_heads[1], drop=drop) for _ in range(depths[1])]
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], 3, (2, 1), 1), nn.GELU()
        )

        self.stage3 = nn.Sequential(
            *[SVTRBlock(dims[2], num_heads[2], drop=drop) for _ in range(depths[2])]
        )

        self.norm = nn.LayerNorm(dims[2])
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.fc   = nn.Linear(dims[2], num_classes)

        if use_frm:
            self.frm = FeatureRearrangement(dims[2])
        if use_sgm:
            self.sgm = SemanticGuidance(dims[2], num_classes)

    def forward(self, x, return_sgm=False):
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.stage1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.merge1(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage2(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.merge2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage3(x)
        x = self.norm(x)

        if self.use_frm:
            x = self.frm(x)

        sgm_out = None
        if self.use_sgm and return_sgm:
            sgm_out = self.sgm(x)

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.pool(x).squeeze(2)
        x = x.permute(0, 2, 1)
        logits = self.fc(x).log_softmax(2)

        if return_sgm and sgm_out is not None:
            return logits, sgm_out
        return logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(variant: str, img_h: int, img_w: int, num_classes: int,
                use_frm=True, use_sgm=True) -> SVTRNet:
    cfg = VARIANTS[variant]
    return SVTRNet(img_h, img_w, num_classes,
                   dims=cfg["dims"], depths=cfg["depths"],
                   num_heads=cfg["num_heads"], drop=cfg["drop"],
                   use_frm=use_frm, use_sgm=use_sgm)


