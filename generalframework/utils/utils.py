import copy
from functools import partial
from pathlib import Path

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

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')


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


def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss_numpy(input, target):
    # with torch.no_grad:
    smooth = 1.
    iflat = input.reshape(input.shape[0], -1)
    tflat = target.reshape(input.shape[0], -1)
    intersection = (iflat * tflat).sum(1)

    foreground_iou = float(
        ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)).mean())

    iflat = 1 - iflat
    tflat = 1 - tflat
    intersection = (iflat * tflat).sum(1)
    background_iou = float(
        ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)).mean())

    return [background_iou, foreground_iou]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.shape[0], -1)
    tflat = target.view(input.shape[0], -1)
    intersection = (iflat * tflat).sum(1)

    foreground_dice = float(
        ((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())

    iflat = 1 - iflat
    tflat = 1 - tflat
    intersection = (iflat * tflat).sum(1)
    background_dice = float(
        ((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())

    return [background_dice, foreground_dice]


def evaluate_dice(val_dataloader, network, save=False):
    b_dice_meter = AverageValueMeter()
    f_dice_meter = AverageValueMeter()
    network.eval()
    with torch.no_grad():
        images = []
        for i, (image, mask, weak_mask, pathname) in enumerate(val_dataloader):
            if mask.sum() == 0 or weak_mask.sum() == 0:
                continue
            image, mask, weak_mask = image.to(device), mask.to(device), weak_mask.to(device)
            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            [b_iou, f_iou] = dice_loss(predicted_mask, mask)
            b_dice_meter.add(b_iou)
            f_dice_meter.add(f_iou)
            if save:
                images = save_images(images, image, proba, mask, weak_mask)
    network.train()
    if save:
        grid = make_grid(images, nrow=4)
        return [[b_dice_meter.value()[0], f_dice_meter.value()[0]], grid]
    else:
        return [[b_dice_meter.value()[0], f_dice_meter.value()[0]], None]


def save_images(images, img, prediction, mask, weak_mask):
    if len(images) >= 30 * 4:
        return images
    segm = pred2segmentation(prediction)
    images.extend([img[0], weak_mask[0].float(), mask[0].float(), segm.float()])
    return images


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


def show_image_mask(*args):
    imgs = [x for x in args if type(x) != str]
    title = [x for x in args if type(x) == str]
    num = len(imgs)
    plt.figure()
    if len(title) >= 1:
        plt.title(title[0])

    for i in range(num):
        plt.subplot(1, num, i + 1)
        try:
            plt.imshow(imgs[i].cpu().data.numpy().squeeze())
        except:
            plt.imshow(imgs[i].squeeze())
    plt.tight_layout()
    plt.show()


def set_boundary_term(g, nodeids, img, kernel_size, lumda, sigma):
    kernel = np.ones((kernel_size, kernel_size))
    kernel[int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)] = 0

    transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma ** 2) * pixel_difference ** 2)
    # =====new =========================================
    padding_size = int(max(kernel.shape) / 2)
    position = np.array(list(zip(*np.where(kernel != 0))))

    def shift_matrix(matrix, kernel):
        center_x, center_y = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
        [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel == 1))))[0]
        dy, dx = kernel_x - center_x, kernel_y - center_y
        shifted_matrix = np.roll(matrix, -dy, axis=0)
        shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
        return shifted_matrix

    for p in position:
        structure = np.zeros(kernel.shape)
        structure[p[0], p[1]] = kernel[p[0], p[1]]
        pad_im = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                        constant_values=0)
        shifted_im = shift_matrix(pad_im, structure)
        weights_ = transfer_function(
            np.abs(pad_im - shifted_im)[padding_size:-padding_size, padding_size:-padding_size])

        g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=False)

    return g


def graphcut_refinement(prediction, image, kernel_size, lamda, sigma):
    '''
    :param prediction: input torch tensor size (batch=1, h, w)
    :param image: input torch tensor size (1, h,w )
    :return: torch tensor long size (1,h,w)
    '''
    prediction_ = prediction.cpu().data.squeeze().numpy()
    prediction_[(prediction_ > 0.5).astype(bool)] = (prediction_[(prediction_ > 0.5).astype(bool)] - 0.5) * 0.5 / (
            prediction_.max() - 0.5) + 0.5
    prediction_[(prediction_ < 0.5).astype(bool)] = (prediction_[(prediction_ < 0.5).astype(bool)] - 0.5) * 0.5 / (
            0.5 - prediction_.min()) + 0.5

    # prediction_ = (prediction_- prediction_.min())/(prediction_- prediction_.min()).max()
    image_ = image.cpu().data.squeeze().numpy()
    unary_term_gamma_1 = 1 - prediction_
    unary_term_gamma_0 = prediction_
    g = maxflow.Graph[float](0, 0)
    # Add the nodes.
    nodeids = g.add_grid_nodes(prediction_.shape)
    g = set_boundary_term(g, nodeids, image_, kernel_size=kernel_size, lumda=lamda, sigma=sigma)
    g.add_grid_tedges(nodeids, (unary_term_gamma_0).squeeze(),
                      (unary_term_gamma_1).squeeze())
    g.maxflow()
    sgm = g.get_grid_segments(nodeids) * 1

    # The labels should be 1 where sgm is False and 0 otherwise.
    new_segmentation = np.int_(np.logical_not(sgm))
    return torch.Tensor(new_segmentation).long().unsqueeze(0)


def graphcut_with_FG_seed_and_BG_dlation(image, weak_mask, full_mask, kernal_size=5, lamda=1, sigma=0.01,
                                         dilation_level=5):
    '''
    :param image: numpy
    :param weak_mask:
    :param full_mask:
    :param kernal_size:
    :param lamda:
    :param sigma:
    :return:
    '''
    if weak_mask.sum() == 0 or full_mask.sum() == 0:
        new_gamma = np.zeros(weak_mask.shape)
    else:
        unary1 = np.zeros(image.squeeze().shape)
        unary0 = np.zeros(unary1.shape)
        unary1[(weak_mask == 1).astype(bool)] = -np.inf
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(weak_mask.astype(np.float32), kernel, iterations=dilation_level)
        unary0[(dilation != 1).astype(bool)] = - np.inf

        g = maxflow.Graph[float](0, 0)
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(image.shape))
        # Add edges with the same capacities.

        # g.add_grid_edges(nodeids, neighbor_term)
        g = set_boundary_term(g, nodeids, image, kernel_size=kernal_size, lumda=lamda, sigma=sigma)

        # Add the terminal edges.
        g.add_grid_tedges(nodeids, (unary0).squeeze(),
                          (unary1).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma = np.int_(np.logical_not(sgm))
    [db, df] = dice_loss_numpy(new_gamma[np.newaxis, :], full_mask[np.newaxis, :])
    return [db, df]


def split_label_unlabel_dataset(train_set, split_ratio):
    np.random.seed(1)
    torch.random.manual_seed(1)
    random_index = np.random.permutation(len(train_set))
    labeled_dataset = copy.deepcopy(train_set)
    labeled_dataset.imgs = [train_set.imgs[x]
                            for x in random_index[:int(len(random_index) * split_ratio)]]
    unlabeled_dataset = copy.deepcopy(train_set)
    unlabeled_dataset.imgs = [train_set.imgs[x]
                              for x in random_index[int(len(random_index) * split_ratio):]]
    return labeled_dataset, unlabeled_dataset


def iter_image_pair(labeled_dataLoader, unlabeled_dataLoader):
    labeled_dataLoader_, unlabeled_dataLoader_ = iter(labeled_dataLoader), iter(unlabeled_dataLoader)
    try:
        labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]
    except:
        labeled_dataLoader_ = iter(labeled_dataLoader)
        labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]

    try:
        unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]
    except:
        unlabeled_dataLoader_ = iter(unlabeled_dataLoader)
        unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]

    return (labeled_img, labeled_mask), (unlabeled_img, unlabeled_mask)


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
from torch import Tensor, einsum
from functools import partial
from typing import Callable, Iterable, List, Set, Tuple, TypeVar

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


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
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy())
