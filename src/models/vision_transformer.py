import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


class MixTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for _ in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for _ in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for _ in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for _ in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        outs.append(rearrange(x, 'b (h w) c -> b c h w', h=H, w=W))

        # stage 2
        x, H, W = self.patch_embed2(outs[-1])
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        outs.append(rearrange(x, 'b (h w) c -> b c h w', h=H, w=W))

        # stage 3
        x, H, W = self.patch_embed3(outs[-1])
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        outs.append(rearrange(x, 'b (h w) c -> b c h w', h=H, w=W))

        # stage 4
        x, H, W = self.patch_embed4(outs[-1])
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        outs.append(rearrange(x, 'b (h w) c -> b c h w', h=H, w=W))

        return outs


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 256, 512], embedding_dim=768, num_classes=4):
        super().__init__()
        assert len(feature_strides) == len(in_channels) == 4

        self.feature_strides = feature_strides
        self.num_classes = num_classes

        self.mlp1 = MLP(input_dim=in_channels[0], embed_dim=embedding_dim)
        self.mlp2 = MLP(input_dim=in_channels[1], embed_dim=embedding_dim)
        self.mlp3 = MLP(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.mlp4 = MLP(input_dim=in_channels[3], embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on multi-level features ############
        n, _, h, w = c1.shape

        _c1 = self.mlp1(c1).transpose(1, 2).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.mlp2(c2).transpose(1, 2).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.mlp3(c3).transpose(1, 2).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c4 = self.mlp4(c4).transpose(1, 2).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))

        x = self.linear_pred(_c)

        return x


class SegFormer(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], decoder_dim=256):
        super().__init__()
        self.encoder = MixTransformer(in_chans=in_channels, embed_dims=embed_dims, num_heads=num_heads,
                                      mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, depths=depths, sr_ratios=sr_ratios)
        self.decoder = SegFormerHead(in_channels=embed_dims, embedding_dim=decoder_dim, num_classes=num_classes)

    def forward(self, x):
        # input x shape: [B, C, H, W]
        h, w = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        # upsample to original size
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits


def build_segformer(device, in_channels=1, classes=4, **kwargs):
    """
    Builds and returns a SegFormer model.
    """
    print(f"Building model: SegFormer with in_channels={in_channels}, classes={classes}")
    model = SegFormer(in_channels=in_channels, num_classes=classes)
    return model.to(device)

