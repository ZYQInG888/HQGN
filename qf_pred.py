import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels=64, out_channels=64, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], in_ch=3):  # in_ch(color:3, gray:1)
        super(ResNet, self).__init__()
        self.in_ch = in_ch
        self.in_channels = 64

        self.conv = nn.Conv2d(self.in_ch, self.in_channels, kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(torch.nn.Linear(512 * block.expansion, 512),
                                 nn.ReLU(),
                                 torch.nn.Linear(512, 512),
                                 nn.ReLU(),
                                 torch.nn.Linear(512, 1)
                                 )

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, channel, downsample=downsample, stride=stride))
        self.in_channels = channel * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channel))

        return nn.Sequential(*layers)

    def forward(self, x):

        h, w = x.size()[-2:]  # [4, 3, 128, 128]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)  # [4, 3, 128, 128]

        out = self.conv(x)  # [4, 64, 64, 64]
        out1 = self.layer1(out)  # [4, 64, 32, 32]
        out2 = self.layer2(out1)  # [4, 128, 16, 16]
        out3 = self.layer3(out2)  # [4, 256, 8, 8]
        out4 = self.layer4(out3)  # [4, 512, 4, 4]

        out = self.avgpool(out4)
        out = torch.flatten(out, 1)
        qf = self.mlp(out)
        qf = nn.Sigmoid()(qf)

        # return qf, out1, out2, out3, out4
        return qf, out1, out2, out3


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


if __name__ == "__main__":
    x = torch.randn(4, 3, 128, 128)  # .cuda()#.to(torch.device('cuda'))
    qf_pred = resnet18()

    path = "./pretrained_model/resnet34-pre.pth"

    try:
        model_dict = qf_pred.state_dict()
        checkpoint = torch.load(path)  # resnet 34 weight
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

        # Initialize the weights of conv layer and the MLP layers
        init.xavier_uniform_(qf_pred.conv.weight)
        pretrained_dict['conv.weight'] = qf_pred.conv.weight

        for i, layer in enumerate(qf_pred.mlp):
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                pretrained_dict[f'mlp.{i}.weight'] = layer.weight
                pretrained_dict[f'mlp.{i}.bias'] = layer.bias

        model_dict.update(pretrained_dict)
        qf_pred.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully.")
    except FileNotFoundError:
        print(f"Pretrained model not found at {path}.")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")

    qf, out1, out2, out3 = qf_pred(x)
    print(qf.shape, out1.shape, out2.shape, out3.shape)