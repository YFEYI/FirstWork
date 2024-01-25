import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from CCCC.cbam import cbam
from typing import Optional, Tuple, Union, Dict
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class CNN_Block(nn.Module):
    def __init__(self,in_C=10,out_C=72,ratio=2,kernel_list=(1,3,5,7),drop=0.):
        super(CNN_Block, self).__init__()
        self.conv11_span = nn.Sequential(
            nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=kernel_list[0], stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[0], stride=1, padding=0),
            nn.ReLU()
        )
        self.conv33_span = nn.Sequential(
            nn.Conv2d(in_channels=out_C,out_channels=out_C,kernel_size=kernel_list[1],stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_C,out_channels=out_C,kernel_size=kernel_list[1],stride=1,padding=1),
            nn.ReLU()
        )
        self.conv55_span = nn.Sequential(
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[2], stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[2], stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv77_span = nn.Sequential(
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[3], stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[3], stride=1, padding=3),
            nn.ReLU(),
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels=out_C*4, out_channels=out_C, kernel_size=kernel_list[0], stride=1, padding=0),
            nn.Conv2d(in_channels=out_C, out_channels=out_C, kernel_size=kernel_list[1], stride=1, padding=1),
            nn.ReLU(),
        )
        self.cbam_after_fusion = cbam(in_size=out_C, reduction_ratio=4, pool_types=['avg', 'max'], kernel_size=3)
    def forward(self,x):
        span11=self.conv11_span(x)
        span33=self.conv33_span(span11)
        span55=self.conv55_span(span33)
        span77 = self.conv77_span(span55)
        span = torch.cat((span11,span33, span55,span77), dim=1)
        x=self.conv_fusion(span)
        x_cbam = self.cbam_after_fusion(x)
        x = x + x_cbam
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, attn_drop=0.,rpe=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rpe = rpe

        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))  # 2*Ph-1 * 2*Pw-1, nH 2*3-1=5 [5*5,head]
            # get pair-wise 相对位置索引 for each token inside one patch
            coords_h = torch.arange(self.patch_size[0])
            coords_w = torch.arange(self.patch_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ph, Pw
            coords_flatten = torch.flatten(coords, 1)  # 2, Ph*Pw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ph*Pw, Ph*Pw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw, Ph*Pw, 2
            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CCBlock(nn.Module):
    def __init__(self, in_c,batch_size=32, size=9,num_heads=None, patch_size=3, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., attn_type="swattn"):
        super(CCBlock, self).__init__()
        self.in_c = in_c  #channel数
        self.num_heads = num_heads  #多头注意力个数
        self.patch_size = patch_size    #窗口大小
        self.mlp_ratio = mlp_ratio      #mlp伸缩倍率
        self.attn_type = attn_type      #使用哪一种注意力

        self.num_patch=size//patch_size
        self.batch_size = batch_size
        self.dropout=nn.Dropout(p=drop)
        self.norm_batch=nn.BatchNorm2d(num_features=in_c)        #有必要Norm吗？
        self.norm_layer = None
        self.attn=None
        self.mlp=None
        if  self.attn_type=='swattn':
            attn_chans=self.in_c
            self.norm_layer=nn.LayerNorm(normalized_shape=attn_chans)
            self.mlp=Mlp(in_features=attn_chans,hidden_features=attn_chans*2,out_features=attn_chans,act_layer=nn.ReLU, drop=drop)
            self.attn = MultiHeadAttention(dim=attn_chans,patch_size=(self.patch_size, self.patch_size),
                                           num_heads=self.num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,rpe=True)
        elif self.attn_type=='crattn':
            attn_chans = self.patch_size**2
            self.norm_layer = nn.LayerNorm(normalized_shape=attn_chans)
            self.mlp = Mlp(in_features=attn_chans, hidden_features=attn_chans * 2, out_features=attn_chans,
                            act_layer=nn.ReLU, drop=drop)
            self.attn = MultiHeadAttention(dim=attn_chans,patch_size=(self.patch_size, self.patch_size),
                                           num_heads=self.num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,rpe=False)
        else:
            attn_chans = self.in_c
            self.norm_layer = nn.LayerNorm(normalized_shape=attn_chans)
            self.mlp = Mlp(in_features=attn_chans, hidden_features=attn_chans * 2, out_features=attn_chans,
                            act_layer=nn.ReLU, drop=drop)
            self.attn = MultiHeadAttention(dim=self.in_c,patch_size=(self.patch_size, self.patch_size),
                                           num_heads=self.num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,rpe=False)
    def unfolding(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
            [B, C, W, H]->[B, P, N, C]
        :param x:
        :return:
        """
        patch_size= self.patch_size
        patch_area = patch_size * patch_size
        batch_size,in_channels,H, W = x.shape

        # number of patches along width and height
        num_patch_w = W // patch_size  # n_w
        num_patch_h = H // patch_size  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B ,C, n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size, in_channels, num_patch_h, patch_size, num_patch_w, patch_size)
        # [B, C, n_h, p_h, n_w, p_w] -> [B, C, n_h, n_w, p_h, p_w]
        x = x.transpose(3, 4)
        # [B, C, n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        info_dict = {
            "orig_size": (H, W),
            "batch_size": batch_size,
            "patch_size":patch_size,
            "patch_area": patch_area,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
        return x, info_dict

    def folding(self, x: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        """
            [B, P, N, C]->[B, H, W, C] 最后还要transpose一下
        :param x:
        :param info_dict:
        :return:
        """
        n_dim = x.dim()
        assert n_dim == 4, "Tensor should be of shape BxPxNxC. Got: {}".format(
            x.shape
        )
        # [B, P, N, C]
        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        patch_area = info_dict["patch_area"]
        patch_size = info_dict["patch_size"]
        # [B, P, N, C]->[B, N, P, C]
        x = x.transpose(1, 2)
        # [B, P, N, C] -> [B, n_h, n_w, p_h, p_w, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, num_patch_h,num_patch_w, patch_size, patch_size, channels)
        # [B, n_h, n_w, p_h, p_w, C]->[B,, n_h, p_h, n_w, p_w, C]
        x = x.transpose(2, 3)
        # [B,, n_h, p_h, n_w, p_w, C]->[B, H, W, C]
        x=x.reshape(batch_size, num_patch_h*patch_size, num_patch_w*patch_size, channels)
        return x
    def forward(self,x):

        B, C, H, W = x.shape#[B,32, 9, 9]
        x, info_dict = self.unfolding(x)  # [B, P, N, C] [B,9,9,32]->[B,9(面积),9(个数),32]
        _, P, N, _ = x.shape  # [B,9,9,32]

        if  self.attn_type=='swattn':
            x=x.transpose(1,2)    #[B, N, P, C]
            x=x.reshape(B*N,P,C)
            shortcut = x
            x=shortcut+self.attn(x)
            # FFN
            x = x + self.mlp(self.norm_layer(x))

            x = x.reshape(B, N, P, C)
            x = x.transpose(1, 2)  # [B, P, N, C]
            x = self.folding(x,info_dict)  # [B,9(面积),9(个数),32]->[B, H, W, C]
            x = x.transpose(1, 2).transpose(1, 3)  # (B, W, H, C)->(B, C, H, W)


        elif self.attn_type=='crattn':
            x = x.transpose(1, 3)  # [B, C, N, P]
            x = x.reshape(B * C, N, P)
            shortcut = x
            x = shortcut+self.attn(x)
            # FFN
            x = x + self.mlp(self.norm_layer(x))

            x = x.reshape(B, C, N, P)
            x = x.transpose(1, 3)  # [B, P, N, C]
            x = self.folding(x,info_dict)  # [B,9(面积),9(个数),32]->[B, H, W, C]
            x = x.transpose(1, 2).transpose(1, 3)  # (B, W, H, C)->(B, C, H, W)

        else:
            x = x.reshape(B*P, N, C)   #[B*P, N, C]
            shortcut = x
            x =shortcut+ self.attn(x)
            # FFN
            x = shortcut + self.mlp(self.norm_layer(x))

            x = x.reshape(B, P, N, C)
            x = self.folding(x,info_dict)  # [B,9(面积),9(个数),32]->[B, H, W, C]
            x = x.transpose(1, 2).transpose(1, 3)  # (B, W, H, C)->(B, C, H, W)
        return x

class CCLayers(nn.Module):
    def __init__(self, in_c,depths,batch_size,num_heads, patch_size=3, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.):
        super(CCLayers, self).__init__()
        self.in_c = in_c  # 动态给予的维度
        self.depths = depths  # 这个layer会重复几次
        self.batch_size=batch_size
        # build blocks
        self.swin_attn_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()
        self.moblie_attn_blocks = nn.ModuleList()
        for i in range(self.depths):
            self.swin_attn_blocks.append(CCBlock(in_c=self.in_c,batch_size=self.batch_size,size=9,num_heads=num_heads,patch_size=patch_size,
                                                 mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop,attn_drop=attn_drop,attn_type='swattn'))

            self.cross_attn_blocks.append(CCBlock(in_c=self.in_c,batch_size=self.batch_size,size=9,num_heads=1,patch_size=patch_size,
                                                 mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop,attn_drop=attn_drop,attn_type='crattn'))

            self.moblie_attn_blocks.append(CCBlock(in_c=self.in_c,batch_size=self.batch_size,size=9,num_heads=num_heads,patch_size=patch_size,
                                                 mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop,attn_drop=attn_drop,attn_type='moattn'))
    def forward(self,x):
        B,H,W,C=x.size()
        num_blocks = len(self.swin_attn_blocks)
        for i in range(num_blocks):
            self.swin_attn_blocks[i].H, self.swin_attn_blocks[i].W = H, W
            self.cross_attn_blocks[i].H, self.cross_attn_blocks[i].W = H, W
            self.moblie_attn_blocks[i].H, self.moblie_attn_blocks[i].W = H, W

            x = self.swin_attn_blocks[i].forward(x)
            x = self.cross_attn_blocks[i].forward(x)
            x = self.moblie_attn_blocks[i].forward(x)
        return x

class CCModule(nn.Module):
    def __init__(self, in_chans=10,num_classes=2,embed_dim=72,batch_size=32, depths=None,
                 num_heads=None,patch_size=3, mlp_ratio=4.,qkv_bias=True,drop=0.1,attn_drop=0.1, ape=True):
        super(CCModule, self).__init__()
        self.in_chans = in_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_ape = ape
        self.patch_size = patch_size
        self.batch_size= batch_size

        self.cnn_block=CNN_Block(in_C=in_chans,out_C=embed_dim,ratio=mlp_ratio,kernel_list=(1,3,5,7))
        # 使用绝对位置偏执
        if self.use_ape:
            patches_resolution = [self.in_chans, self.in_chans]
            self.ape = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.ape, std=.02)  # 初始化绝对位置偏执
        # stochastic depth 动态的drop率
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            attn_layers = CCLayers(in_c=embed_dim,depths=depths[i_layer],batch_size=batch_size,num_heads=num_heads[i_layer],patch_size=patch_size,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop,attn_drop=attn_drop)
            self.layers.append(attn_layers)
        #self.norm = nn.LayerNorm(normalized_shape=num_classes)
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.softmax=nn.Softmax(dim=-1)

        self.fusion=nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2,out_channels=embed_dim,kernel_size=(1,1),stride=1,padding=1),
            nn.ReLU(),
        )

        self.last_mlp=Mlp(in_features=self.embed_dim*81,hidden_features=1024,out_features=512,act_layer=nn.ReLU,drop=0.5)
        self.head=nn.Linear(in_features=512 ,out_features=num_classes,bias=True)

    def forward(self,x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        x = self.cnn_block(x)
        if self.use_ape:
            # interpolate the absolute position encoding to the corresponding size
            #interpolate上下采样函数，mode=bicubic指使用的是双三次插值法
            ape = F.interpolate(self.ape, size=(H, W), mode='bicubic')
            # 插入绝对位置编码???只是加了进去没有改变形状
            x = x + ape  # B Wh*Ww C  [B,32,9,9]+[B,1,9,9]
        shortcut = x

        for layer in self.layers:
            x = layer(x)
        #x = self.avgpool(x)  #ver 7.0 将此删除，先进行瓶颈结构后，展平最后进行分类
        x=torch.cat((shortcut,x),dim=1)
        x=self.fusion(x)
        x = torch.flatten(x, 1)
        x=self.last_mlp(x)
        x=self.head(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU, drop=0.):
       super(Mlp, self).__init__()
       out_features=out_features or in_features
       hidden_features=hidden_features or in_features

       self.fc1=nn.Linear(in_features,hidden_features)
       self.act=act_layer()
       self.drop1 = nn.Dropout(drop)
       self.fc2 = nn.Linear(hidden_features, out_features)
       self.drop2 = nn.Dropout(drop)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def CCCA_small(num_classes: int = 2, **kwargs):
    model = CCModule(in_chans=5,num_classes=num_classes,embed_dim=36,batch_size=64,depths=[1],num_heads=[2],patch_size=3,mlp_ratio=4,qkv_bias=True
                     ,drop=0.,attn_drop=0.,ape=True)
    return model

def CCCA_base(num_classes: int = 2, **kwargs):
    model = CCModule(in_chans=5,num_classes=num_classes,embed_dim=72,batch_size=64,depths=[1],num_heads=[2],patch_size=3,mlp_ratio=4,qkv_bias=True
                     ,drop=0.,attn_drop=0.,ape=True)
    return model

def CCCA_large(num_classes: int =2, **kwargs):
    model = CCModule(in_chans=5,num_classes=num_classes,embed_dim=144,batch_size=64,depths=[1],num_heads=[2],patch_size=3,mlp_ratio=4,qkv_bias=True
                     ,drop=0.,attn_drop=0.,ape=True)
    return model


if __name__ == '__main__':
    model=CCCA_small()
    print(model)