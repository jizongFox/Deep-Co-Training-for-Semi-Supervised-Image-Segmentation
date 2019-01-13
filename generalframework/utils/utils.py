import copy
from functools import partial
from pathlib import Path
from torch import Tensor, einsum
from functools import partial
from typing import Callable, Iterable, List, Set, Tuple, TypeVar

import cv2
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imsave
from torchnet.meter import AverageValueMeter
from torchvision.utils import make_grid
from tqdm import tqdm
import warnings

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

tqdm_ = partial(tqdm, ncols=175,
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


class AverageMeter(object):
    """Computes and stores the average and current value for scalar """

    def __init__(self):
        """ Constructor """
        self.reset()

    def reset(self):
        """ Reset stats """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update the stats assuming `val` is accumulated `n` times """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    assert simplex(probs)

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


def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy())


## dataset
from torch.utils.data import DataLoader


class iterator_(object):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.iter_dataloader = iter(dataloader)

    def __next__(self):
        try:
            return self.iter_dataloader.__next__()
        except:
            self.iter_dataloader = iter(self.dataloader)
            return self.iter_dataloader.__next__()
