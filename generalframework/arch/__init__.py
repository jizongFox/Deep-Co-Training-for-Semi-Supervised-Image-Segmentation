""" Define the architecture. Modified from torchvision
"""
from .network import FCN8, FCN16, FCN32, UNet, SegNet, PSPNet, UNet_bn
from .enet import Enet
from .deeplab import DeepLabV2, DeepLabV3, DeepLabV3Plus
from .joseent import ENet as JEnet
from .joseent import CorstemNet as CNet
from torch import nn
from .deeplab.enet import ENet as DEnet
from .pretrain_torchvision import vgg_11, resnet_18

__all__ = ['weights_init', 'get_arch']
"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES = {}


def _register_arch(arch, callable, alias=None):
    """ Private method to register the architecture to the ARCH_CALLABLES
        :param arch: A str
        :param callable: The callable that return the nn.Module
        :param alias: None, or a list of string, or str
    """
    if arch in ARCH_CALLABLES:
        raise ValueError('{} already exists!'.format(arch))
    ARCH_CALLABLES[arch] = callable
    if alias:
        if isinstance(alias, str):
            alias = [alias]
        for other_arch in alias:
            if other_arch in ARCH_CALLABLES:
                raise ValueError('alias {} for {} already exists!'.format(other_arch, arch))
            ARCH_CALLABLES[other_arch] = callable


# Adding architecture (new architecture goes here...)
_register_arch('fcn8', FCN8)
_register_arch('fcn16', FCN16)
_register_arch('fcn32', FCN32)
_register_arch('unet', UNet)
_register_arch('unet_bn', UNet_bn)
_register_arch('segnet', SegNet)
_register_arch('enet', Enet)
_register_arch('jenet', JEnet)
_register_arch('cnet', CNet)
_register_arch('deeplabv2', DeepLabV2)
_register_arch('deeplabv3', DeepLabV3)
_register_arch('deeplabv3plus', DeepLabV3Plus)
_register_arch('deeplabenet', DEnet)
_register_arch('vgg11', vgg_11)
_register_arch('resnet18', resnet_18)

"""
Public interface
"""


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_arch(arch, kwargs) -> Enet:
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch)
    try:
        kwargs.pop('arch')
    except:
        pass
    assert arch_callable, "Architecture {} is not found!".format(arch)
    net = arch_callable(**kwargs)
    try:
        net.apply(weights_init)
    except AttributeError as e:
        print(f'Using pretrained models with the error:{e}')
    return net
