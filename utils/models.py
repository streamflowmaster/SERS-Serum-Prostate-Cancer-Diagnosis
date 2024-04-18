import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


def conv1d_5(inplanes, outplanes, stride=1):

    return nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
# 下采样block，下采样过程中使用resnet作为backbone
class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, bn=False):
        super(Block, self).__init__()
        self.bn = bn
        self.conv1 = conv1d_5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv1d_5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# 上采样block
class Decoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=5, stride=5):
        super(Decoder_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                           padding=0, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1 = conv1d_5(inplanes, outplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv1d_5(outplanes, outplanes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        out = torch.cat((x1, x2), dim=1)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out


class Resnet1d(nn.Module):

    def __init__(self, inplanes=5, layers=[2,2,2,2]):
        super(Resnet1d, self).__init__()
        self.inplanes = inplanes
        self.encoder1 = self._make_encoder(Block, 32, layers[0], 5)
        self.encoder2 = self._make_encoder(Block, 64, layers[1], 5)
        self.encoder3 = self._make_encoder(Block, 128, layers[2], 5)
        self.encoder4 = self._make_encoder(Block, 256, layers[3], 4)
        self.decoder3 = Decoder_block(256, 128, stride=4, kernel_size=4)
        self.decoder2 = Decoder_block(128, 64)
        self.decoder1 = Decoder_block(64, 32)
        # 如果要修改输出的通道，在这里修改
        self.conv1x1 = nn.ConvTranspose1d(32, 1, kernel_size=5, stride=5, bias=False)

    def _make_encoder(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes or stride != 1:
            downsample = nn.Conv1d(self.inplanes, planes, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.encoder2(down1)
        down3 = self.encoder3(down2)
        down4 = self.encoder4(down3)

        up3 = self.decoder3(down4, down3)
        up2 = self.decoder2(up3, down2)
        up1 = self.decoder1(up2, down1)
        out = self.conv1x1(up1)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_without_embed(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet_without_embed, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(6144, num_classes)

        # 遍历所有模块，然后对其中参数进行初始化
        for m in self.modules():  # self.modules()采用深度优先遍历的方式，存储了net的所有模块
            if isinstance(m, nn.Conv2d):  # 判断是不是卷积
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 对权值参数初始化
            elif isinstance(m, nn.BatchNorm2d):  # 判断是不是数据归一化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 当是 3 4 6 3 的第一层时，由于跨层要做一个步长为 2 的卷积 size会变成二分之一，所以此处跳连接 x 必须也是相同维度
            downsample = nn.Sequential(  # 对跳连接 x 做 1x1 步长为 2 的卷积，保证跳连接的时候 size 一致
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 将跨区的第一个要做步长为 2 的卷积添加到layer里面
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 将除去第一个的剩下的 block 添加到layer里面
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0,1,3,2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_(self,x):
        x = x.permute(0, 1, 3, 2)
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet50_without_embed_(pretrained=True,num_classes=3):
    '''Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''

    model = ResNet_without_embed(Bottleneck, [3, 4, 6, 3],  num_classes=num_classes)
    if pretrained:  # 加载已经生成的模型
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def resnet18_without_embed_(pretrained=True, num_classes=3):
    '''Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''

    model = ResNet_without_embed(Bottleneck, [2,2,2,2], num_classes=num_classes)
    if pretrained:  # 加载已经生成的模型
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


