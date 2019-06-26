from torchvision.models import vgg16, vgg11, resnet18
from torch import nn


def vgg_16(pretrain=True, num_classes=1):
    net = vgg16(pretrain)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def vgg_11(pretrain=True, input_channel=3, num_classes=1):
    net = vgg11(pretrain)
    if input_channel != 3:
        net.features[0] = nn.Conv2d(
            in_channels=input_channel,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def resnet_18(pretrain=True, input_channel=3, num_classes=1):
    net = resnet18(pretrain)
    if input_channel != 3:
        net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    net.fc = nn.Linear(512, num_classes)
    return net


if __name__ == '__main__':
    resnet_18(True)
