import math
from turtle import st
import numpy as np

import megengine as mge
from megengine import module as M
from megengine import functional as F


##########################################################################
# Restormer
##########################################################################

class FeedForward(M.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = M.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = M.Conv2d(hidden_features*2, hidden_features*2,
                               kernel_size=5, stride=1,
                               padding=4, dilation=2,
                               groups=hidden_features*2, bias=bias)

        self.project_out = M.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = F.split(self.dwconv(x), 2, axis=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(M.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = mge.Parameter(F.ones((num_heads, 1, 1)))

        self.qkv = M.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = M.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = M.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.softmax = M.Softmax(axis=-1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = F.split(qkv, 3, axis=1)
        q = q.reshape(b, self.num_heads, c//self.num_heads, -1)
        k = k.reshape(b, self.num_heads, c//self.num_heads, -1)
        v = v.reshape(b, self.num_heads, c//self.num_heads, -1)

        q = F.normalize(q, axis=-1)
        k = F.normalize(k, axis=-1)

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.temperature
        attn = self.softmax(attn)

        out = F.matmul(attn, v)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class LayerNorm(M.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mge.Parameter(
            np.ones((1, channels, 1, 1), dtype='float32'))
        self.bias = mge.Parameter(
            np.zeros((1, channels, 1, 1), dtype='float32'))

    def forward(self, x):
        mu = F.mean(x, axis=1, keepdims=True)
        sigma = F.var(x, axis=1, keepdims=True)
        return (x - mu) / F.sqrt(sigma + self.eps) * self.weight + self.bias


class TransformerBlock(M.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
# Fusion modules
##########################################################################

class PDConvFuse(M.Module):
    def __init__(self, in_channels, feature_num=2, bias=True):
        super().__init__()
        self.feature_num = feature_num

        self.pwconv = M.Conv2d(feature_num*in_channels,
                               in_channels, 1, 1, 0, bias=bias)
        self.dwconv = M.Conv2d(in_channels, in_channels,
                               3, 1, 1, bias=bias, groups=in_channels)

    def forward(self, *inp_feats):
        return self.dwconv(F.gelu(self.pwconv(F.concat(inp_feats, axis=1))))


class SimpleGateFuse(M.Module):
    def __init__(self, in_channels, feature_num=2, bias=True):
        super().__init__()
        hidden_channel = 2 * in_channels
        self.conv1x1 = M.Conv2d(in_channels*feature_num,
                                hidden_channel, 1, 1, 0, bias=bias)

    def forward(self, inp_feats):
        x = self.conv1x1(F.concat(inp_feats, axis=1))
        x1, x2 = F.split(x, 2, 1)
        return F.gelu(x2) * x1


##########################################################################
# Feedback block
##########################################################################


class FeedbackBlock(M.Module):
    def __init__(self,
                 num_feats,
                 num_blocks,
                 num_refine_feats,
                 num_reroute_feats
                 ):
        super(FeedbackBlock, self).__init__()

        self.num_refine_feats = num_refine_feats
        self.num_reroute_feats = num_reroute_feats
        self.num_blocks = num_blocks

        self.trans_list = [
            TransformerBlock(dim=32, num_heads=4, ffn_expansion_factor=1.2)
            for _ in range(num_blocks)
        ]

        self.skip_blocks = [
            PDConvFuse(num_feats, feature_num=2)
            for _ in range(math.ceil(self.num_blocks / 2) - 1)
        ]

        self.GFM = M.Sequential(
            SimpleGateFuse(num_feats, feature_num=num_reroute_feats),
            M.Conv2d(num_feats, num_feats, 1, 1, 0)
        )

    def forward(self, input_feat, last_feats_list):

        cur_feats_list = []
        skip_features = []

        idx_skip = 1
        for idx, b in enumerate(self.trans_list):
            # refining the lowest-level features
            if len(last_feats_list) > 0 and idx < self.num_refine_feats:
                input_feat = self.GFM(input_feat + last_feats_list)

            if idx > math.floor(self.num_blocks / 2):
                input_feat = self.skip_blocks[idx_skip -
                                              1](input_feat, skip_features[-idx_skip])
                idx_skip += 1

            input_feat = b(input_feat)
            cur_feats_list.append(input_feat)

            if idx < (math.ceil(self.num_blocks / 2) - 1):
                skip_features.append(input_feat)

        # rerouting the highest-level features
        return cur_feats_list[-self.num_reroute_feats:]


class FeedbackRestormer(M.Module):
    def __init__(
        self,
        in_channels=4,
        num_feats=32,
        num_blocks=7,
        num_steps=2,
        num_refine_feats=1,
        num_reroute_feats=3,
    ):
        super().__init__()

        mean = mge.tensor(np.array([0.08958147, 0.11184454, 0.11297596, 0.09162903]).reshape((1,4,1,1)), dtype='float32')
        std = mge.tensor(np.array([0.06037381, 0.09139108, 0.09338845, 0.06839059]).reshape((1,4,1,1)), dtype='float32')
        self.mean = mge.Parameter(mean, is_const=True)
        self.std = mge.Parameter(std, is_const=True)

        self.num_steps = num_steps
        self.num_blocks = num_blocks

        self.conv_in = M.Sequential(
            M.Conv2d(in_channels, num_feats, 5, 1, 2),
            M.Conv2d(num_feats, num_feats, 5, 1, 2, groups=num_feats),
            M.Conv2d(num_feats, num_feats, 1, 1, 0)
        )

        self.conv_out = M.Sequential(
            M.Conv2d(num_feats, num_feats, 5, 1, 2, groups=num_feats),
            M.Conv2d(num_feats, num_feats, 1, 1, 0),
            M.Conv2d(num_feats, in_channels, 5, 1, 2)
        )

        self.block = FeedbackBlock(
            num_feats,
            num_blocks,
            num_refine_feats,
            num_reroute_feats
        )

    def norm(self, x):
        return (x - self.mean) / self.std

    def renorm(self, x):
        return x * self.std + self.mean

    def forward(self, x):
        x = self.norm(x)
        shortcut = x
        init_feat = self.conv_in(x)
        x_list = []
        last_feats_list = []

        for _ in range(self.num_steps):
            last_feats_list = self.block(init_feat, last_feats_list)
            x_list.append(self.conv_out(last_feats_list[-1]) + shortcut)

        return self.renorm(x_list[-1])
