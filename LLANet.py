import torch
import torch.nn.functional as F
from utils import *
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from typing import Optional, Callable
from Window_attention import WindowAttention
from einops import rearrange, repeat
from torchsummary import summary



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class LLA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        ### DOR-MLP
           ### OR-MLP

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

           ### OR-MLP
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)
        x2 = x.transpose(1, 2).view(B, C, H, W)
        
        ### DSC
        x2 = self.dwconv(x2, H, W)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)
        
        x3 = torch.cat([x1, x2], dim=2)
        x3 = self.fc6(x3)
        x3 = self.drop(x3)
        return x3


class LLABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LLA(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.drop_path(self.mlp(x, H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)
        self.relu=nn.ReLU()
        self.bch=nn.BatchNorm2d(dim)
        self.local=WindowAtten(dim)
    def forward(self, x, H, W):
        x=self.local(x)
        x=self.relu(self.bch(x))
        return x


class Feature_Incentive_Block(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv1= nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv15=nn.Conv2d(in_ch,out_ch,1)
        self.bat1=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)
        self.conv2= nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bat2= nn.BatchNorm2d(out_ch)

    def forward(self, input):
        input=self.relu(self.bat1(self.conv1(input)))
        t=input
        input=self.relu(self.relu(self.bat2(self.conv2(input)))+t)
        return input


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class LLANet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[64, 128, 256, 512, 1024],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv111=nn.Conv2d(3,embed_dims[0]//2,1)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[3],
                                                    embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[4],
                                                    embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([LLABlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([LLABlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.1, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([LLABlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])
        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 8)
        self.rfb=MSF(embed_dims[0],8,embed_dims[0])
        self.rfb2=MSF(embed_dims[1],16,embed_dims[1])
        self.rfb3=MSF(embed_dims[2],32,embed_dims[2])
        self.rfb4=MSF(embed_dims[3],64,embed_dims[3])
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)
        
    def forward(self, x):
        B = x.shape[0]
        
        ### Conv Stage
       

        out = self.conv1(x)
        t1=self.rfb(out)
        out = self.pool1(out)
      
        out = self.conv2(out)
        t2=self.rfb2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        t3=self.rfb3(out)
        out = self.pool3(out)
       
        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.rfb4(out)
        #t4=out
        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        out = torch.add(out, t3)
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
     
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MSF(nn.Module):
    def __init__(self, channels_in=1024, channels_mid=128, channels_out=32):
        super(MSF, self).__init__()
        self.global_se = SELayer(channels_in)
        self.reduce = nn.Sequential(nn.Conv2d(channels_in, channels_mid, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(channels_mid),
                                    nn.PReLU(channels_mid))
        self.br0 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=1, bias=False,
                      bn=True, relu=True),
            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=1, padding=1, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br1 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=1, padding=1, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=3, padding=3, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br2 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=5, dilation=1, padding=2, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=5, padding=5, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.br3 = nn.Sequential(
            BasicConv(channels_mid, channels_mid, kernel_size=7, dilation=1, padding=3, groups=channels_mid, bias=False,
                      bn=True, relu=False),
            BasicConv(channels_mid, channels_mid, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(channels_mid, channels_mid, kernel_size=3, dilation=7, padding=7, groups=channels_mid, bias=False,
                      relu=False),
        )
        self.point_global = BasicConv(channels_mid * 4 + channels_in, channels_out, kernel_size=1, bias=False, bn=True,
                                      relu=True)

    def forward(self, x):
        x_reduce = self.reduce(self.global_se(x))
        x0 = self.br0(x_reduce)
        x1 = self.br1(x_reduce)
        x2 = self.br2(x_reduce)
        x3 = self.br3(x_reduce)
        out = self.point_global(torch.cat([x, x0, x1, x2, x3], dim=1))
        return out

    
class WindowAtten(nn.Module):
    def __init__(self,input_chanl):
        super(WindowAtten,self).__init__()
        self.q=nn.Conv2d(input_chanl,input_chanl,1)
        self.k=nn.Conv2d(input_chanl,input_chanl,1)
        self.v=nn.Conv2d(input_chanl,input_chanl,1)
        self.soft=nn.Softmax(2)
        self.gama=nn.Parameter(torch.zeros(1))
        self.locatt=WindowAttention(7,False,1)
    def forward(self,x):
        x_q=self.q(x)
        x_k=self.k(x)
        x_v=self.v(x)
        x_end=self.locatt(x_q,x_k,x_v)

        return x_end


if __name__ == "__main__":
    model=LLANet(3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    