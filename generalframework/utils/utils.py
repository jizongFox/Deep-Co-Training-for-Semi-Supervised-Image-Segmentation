import argparse
import collections
import os
import random
import sys
import warnings
from copy import deepcopy as dcopy
from functools import partial
from functools import reduce
from pathlib import Path
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imsave
from torch import Tensor, einsum
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def pred2class(pred: torch.Tensor) -> Tensor:
    '''
    return the class prediction whether for pred_logit or pred_after_softmax
    :param pred: input Tensor of B,C,W,H
    :return: B,W,H
    '''
    assert pred.shape.__len__() == 4, pred.shape
    return pred.max(1)[1]


# fns

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def soft_size(a: Tensor) -> Tensor:
    '''
    Soft some of each class per image
    :param a:
    :return:
    '''
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    '''
    Soft sum of each class among images
    :param a:
    :return:
    '''
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    '''
    todo: understand this function
    :param a:
    :return:
    '''
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


def simplex(t: Tensor, axis=1) -> bool:
    '''
    check if the matrix is the probability
    :param t:
    :param axis:
    :return:
    '''
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    '''
    check if the Tensor is Onehot
    :param t:
    :param axis:
    :return:
    '''
    return simplex(t, axis) and sset(t, [0, 1])


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
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res
def predlogit2one_hot(logit: Tensor) -> Tensor:
    _, C, _, _ = logit.shape
    probs = F.softmax(logit,1)
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res


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


def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, seg_num=None) -> None:
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
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


## argparser

def yaml_parser() -> dict:
    parser = argparse.ArgumentParser('Augment parser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])

    args: argparse.Namespace = parser.parse_args()
    args: dict = _parser(args.strings)
    # pprint(args)
    return args


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


## dictionary functions
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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


def extract_from_big_dict(big_dict, keys) -> dict:
    """ Get a small dictionary with key in `keys` and value
        in big dict. If the key doesn't exist, give None.
        :param big_dict: A dict
        :param keys: A list of keys
    """
    #   TODO a bug has been found
    return {key: big_dict.get(key) for key in keys if big_dict.get(key, 'not_found') != 'not_found'}


## search path functions
def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)]


# taken from mean teacher paper


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
