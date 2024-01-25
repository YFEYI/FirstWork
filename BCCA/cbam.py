import torch
import torch.nn as nn
import torch.nn.functional as F

class cbam(nn.Module):
    def __init__(self,in_size,reduction_ratio=16,pool_types=['avg','max'],no_spatial=False,kernel_size=7):
        """
            in_size:MLP输入尺寸
            reduction_ration:MLP隐藏层缩放比率
            pool_types:max or avg pool
            no_spatial:是否有空间注意力
        """
        super(cbam, self).__init__()
        self.ChannelGate=ChannelGate(in_size,reduction_ratio,pool_types)
        self.spatial=no_spatial
        #if not no_spatial:
        self.SpatialGate=SpatialGate(kernel_size=kernel_size)
        pass
    def forward(self,x):
        x_cout=self.ChannelGate(x)
        #if not self.no_spatial:
        x_scout = self.SpatialGate(x_cout)
        return x_scout


class ChannelGate(nn.Module):
    """
        通道注意力
        输入：R^C*H*W
        maxpool:7*7/avgpool:7*7
        MLP
        Mc(F)=signmoid(MLP(avgpool(F))+MLP(maxpool(F)))
    """
    def __init__(self,in_size,reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.in_size=in_size
        self.mlp=nn.Sequential(
            Flatten(),
            nn.Linear(in_size,in_size//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_size//reduction_ratio,in_size)
        )
        self.pool_types=pool_types

    def forward(self,x):
        """
            Mc(F)=signmoid(MLP(avgpool(F))+MLP(maxpool(F)))
            channel_att_sum:组合avg和max
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                Favg=F.avg_pool2d(x,(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                mlpFavg=self.mlp(Favg)
            elif pool_type=='max':
                Fmax=F.max_pool2d(x,(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                mlpFmax=self.mlp(Fmax)

            if channel_att_sum is None:
                channel_att_sum=mlpFavg
            else:
                channel_att_sum+=mlpFmax
        scale=torch.sigmoid(channel_att_sum)
        scale=scale.reshape(scale.shape[0],-1,1,1)
        return x*scale


class SpatialGate(nn.Module):
    """
        空间注意力
        输入：R^C*H*W
        maxpool:7*7/avgpool:7*7
        MLP
        Ms(F)=signmoid(f^7([avgpool(F);maxpool(F)]))
    """

    def __init__(self,kernel_size):
        super(SpatialGate, self).__init__()
        self.kernel_size=kernel_size #7
        self.FFpool=ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
    def forward(self,x):
        x_ff = self.FFpool(x)
        x_out = self.spatial(x_ff)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):
    def forward(self, x):
        """
        我怎么感觉max(x,dim=0?)
        """
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x