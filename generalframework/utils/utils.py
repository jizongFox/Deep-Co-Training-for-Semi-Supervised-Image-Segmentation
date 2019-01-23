import warnings
import os
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, Union, Any
from pprint import pprint
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.io import imsave
from torch import Tensor, einsum
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import argparse
import collections

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

tqdm_ = partial(tqdm, ncols=125,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
        # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:

                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print(1)
        return color_image


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


def pred2class(pred: torch.Tensor):
    assert pred.shape.__len__() == 4, pred.shape
    return pred.max(1)[1]


class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


def extract_from_big_dict(big_dict, keys):
    """ Get a small dictionary with key in `keys` and value
        in big dict. If the key doesn't exist, give None.
        :param big_dict: A dict
        :param keys: A list of keys
    """
    #   TODO a bug has been found
    return {key: big_dict.get(key) for key in keys if big_dict.get(key, 'not_found') != 'not_found'}


# fns

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", a).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


# check if the matrix is the probability
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs, 1)
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    try:
        assert simplex(probs)
    except:
        import ipdb
        ipdb.set_trace()
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, seg_num=None) -> None:
    b, w, h = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            if seg_num is None:
                save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
            else:
                save_path = Path(root, f"iter{iter:03d}", mode, seg_num, name).with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy())


## dataset


class iterator_(object):
    def __init__(self, dataloader: Union[DataLoader, List[Any]]) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.iter_dataloader = iter(dataloader)

    def __next__(self):
        try:
            return self.iter_dataloader.__next__()
        except:
            self.iter_dataloader = iter(self.dataloader)
            return self.iter_dataloader.__next__()


## adversarial generation
def adversarial_fgsm(image: Tensor, data_grad: Tensor, epsilon: float = 0.001):
    """
    FGSM for generating adversarial sample
    :param image: original clean image
    :param target: original clean target
    :param epsilon: the pixel-wise perturbation amount
    :param data_grad: gradient of the loss w.r.t the input image
    :return: perturbed image representing adversarial sample
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class fsgm_img_generator(object):

    def __init__(self, net: nn.Module, eplision: float = 0.5) -> Tensor:
        super().__init__()
        self.net = net
        self.eplision = eplision

    def __call__(self, img: Tensor, gt: Tensor, criterion: nn.Module):
        tra_state = self.net.training
        self.net.eval()
        img.requires_grad = True
        self.net.zero_grad()
        pred = self.net(img)
        loss = criterion(pred, gt.squeeze(1))
        loss.backward()
        adv_img = adversarial_fgsm(img, img.grad, epsilon=self.eplision).detach()
        if tra_state == True:
            self.net.train()
        assert self.net.training == tra_state
        return adv_img.detach()


class VATGenerator(object):
    def __init__(self, net: nn.Module, xi=1e-6, eplision=10, ip=1):
        """VAT generator based on https://arxiv.org/abs/1704.03976
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eplision
        self.ip = ip
        self.net = net

    @staticmethod
    def _l2_normalize(d):
        # d = d.cpu().numpy()
        # d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        # return torch.from_numpy(d)
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    @staticmethod
    def kl_div_with_logit(q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)

        return qlogq - qlogp

    def __call__(self, img):
        tra_state = self.net.training
        self.net.eval()
        with torch.no_grad():
            pred = self.net(img)

        # prepare random unit tensor
        d = torch.Tensor(img.size()).normal_()  # 所有元素的std =1, average = 0
        d = self._l2_normalize(d).to(img.device)
        self.net.zero_grad()
        for _ in range(self.ip):
            d = self.xi * self._l2_normalize(d).to(img.device)
            d.requires_grad = True
            y_hat = self.net(img + d)
            delta_kl = self.kl_div_with_logit(pred.detach(), y_hat).mean()
            delta_kl.backward()

            d = d.grad.data.clone().cpu()
            self.net.zero_grad()
        ##
        d = self._l2_normalize(d).to(img.device)
        r_adv = self.eps * d
        # compute lds
        img_adv = img + r_adv.detach()
        if tra_state == True:
            self.net.train()
        assert self.net.training == tra_state

        return img_adv.detach()


## argparser

def yaml_parser() -> dict:
    parser = argparse.ArgumentParser('Augment parser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])

    args: argparse.Namespace = parser.parse_args()
    args: dict = _parser(args.strings)
    # pprint(args)
    return args


from copy import deepcopy as dcopy
from functools import reduce


def _parser(strings: List[str]) -> List[dict]:
    assert isinstance(strings, list)
    ## no doubled augments
    assert set(map_(lambda x: x.split('=')[0], strings)).__len__() == strings.__len__(), 'Augment doubly input.'
    args: List[dict] = [_parser_(s) for s in strings]
    args = reduce(lambda x, y: dict_merge(x, y, True), args)
    return args


def _parser_(input_string: str) -> Union[dict, None]:
    if input_string.__len__() == 0:
        return None
    assert input_string.find('=') > 0, f"Input args should include '=' to include the value"
    keys, value = input_string.split('=')[:-1][0].replace(' ', ''), input_string.split('=')[1].replace(' ', '')
    keys = keys.split('.')
    keys.reverse()
    for k in keys:
        d = {}
        d[k] = value
        value = dcopy(d)
    return dict(value)


def dict_merge(dct: dict, merge_dct: dict, re=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # dct = dcopy(dct)
    if merge_dct is None:
        if re:
            return dct
        else:
            return
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            try:
                dct[k] = type(dct[k])(eval(merge_dct[k])) if type(dct[k]) in (bool, list) else type(dct[k])(
                    merge_dct[k])
            except:
                dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
