import torch.nn as nn
import torch
"""
    resnet34
"""
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channel,out_channel,nodownsample=None,width_per_group=72):
        super(BasicBlock, self).__init__()
        self.nodownsample = nodownsample
        self.conv1=nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                  kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        identity = x
        if self.nodownsample is not None:
            identity = self.nodownsample(x)
            x = identity

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=5,
                 include_top=True,groups=1,width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 72

        self.groups = groups
        self.width_per_group = width_per_group
        # 取消ResNet，第一层的下采样
        self.conv1 = nn.Conv2d(5 , self.in_channel, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 不带池化层玩
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 72, blocks_num[0],down=False)
        self.layer2 = self._make_layer(block, 72, blocks_num[1], down=False)
        self.layer3 = self._make_layer(block, 144, blocks_num[2], down=True)
        #self.layer4 = self._make_layer(block, 144, blocks_num[3], down=False)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def _make_layer(self, block, channel, block_num, down=False):
        nodownsample = None
        if down and self.in_channel!=channel * block.expansion:
            nodownsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
            self.in_channel = channel * block.expansion

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            nodownsample=nodownsample,
                            width_per_group=self.width_per_group))
        for i in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

def resnet34(num_classes=5, include_top=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6], num_classes=num_classes, include_top=include_top)

if __name__ == '__main__':
    model=resnet34()
    print(model)
