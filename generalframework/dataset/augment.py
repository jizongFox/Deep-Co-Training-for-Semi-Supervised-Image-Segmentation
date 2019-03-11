import collections
import math
import numbers
import random

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps
from scipy.ndimage import rotate
from skimage.transform import resize
from torch import Tensor
from torchvision import transforms


class temporary_seed:
    def __init__(self, random_seed, np_seed):
        self.random_seed = random_seed
        self.np_seed = np_seed

    def __enter__(self):
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()
        np.random.set_state(self.np_seed)
        random.setstate(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)


class ToLabel():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img):
        np_img = np.array(img)[None, ...]
        t_img = torch.from_numpy(np_img)
        return t_img.long()


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask)
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            tf.affine(mask,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.NEAREST,
                      fillcolor=255,
                      shear=0.0))


class Scale(object):
    def __init__(self, size):
        if isinstance(size, str):
            size = eval(size)
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return img, mask
        if w > h:
            ow = self.size[1]
            oh = int(self.size[0] * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size[0]
            ow = int(self.size[1] * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, mask))


key2aug = {
    "rcrop": RandomCrop,
    "scale": Scale,
    "rsize": RandomSized,
    "rsizecrop": RandomSizedCrop,
    "rotate": RandomRotate,
    "ccrop": CenterCrop,
    "sale": Scale
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        print("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        print("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


def PILaugment(img_list):
    if random.random() > 0.5:
        img_list = [ImageOps.flip(img) for img in img_list]
    if random.random() > 0.5:
        img_list = [ImageOps.mirror(img) for img in img_list]
    if random.random() > 0.5:
        angle = random.random() * 90 - 45
        img_list = [img.rotate(angle, resample=Image.NEAREST) for img in img_list]

    if random.random() > 0.5:
        (w, h) = img_list[0].size
        crop = random.uniform(0.85, 0.95)
        W = int(crop * w)
        H = int(crop * h)
        start_x = w - W
        start_y = h - H
        x_pos = int(random.uniform(0, start_x))
        y_pos = int(random.uniform(0, start_y))
        img_list = [img.crop((x_pos, y_pos, x_pos + W, y_pos + H)) for img in img_list]

    return img_list


def TensorAugment_4_dim(img_list):
    img_list = [img.cpu().numpy() for img in img_list] if isinstance(img_list[0], Tensor) else img_list
    if random.random() > 0.5:
        img_list = [np.flip(img, axis=1) for img in img_list]
    if random.random() > 0.5:
        img_list = [np.flip(img, axis=2) for img in img_list]
    if random.random() > 0.5:
        angle = random.random() * 90 - 45
        img_list = [rotate(img, axes=(1, 2), angle=angle, reshape=False, mode='constant', prefilter=True, order=4) for
                    img in img_list]
    if random.random() > 0.5:
        (w, h) = img_list[0][0].shape
        crop = random.uniform(0.85, 0.95)
        W = int(crop * w)
        H = int(crop * h)
        start_x = w - W
        start_y = h - H
        x_pos = int(random.uniform(0, start_x))
        y_pos = int(random.uniform(0, start_y))
        img_list = [img[:, y_pos:y_pos + H, x_pos:x_pos + W, ] for img in img_list]

    img_list = [resize(img, output_shape=(img.shape[0], 256, 256), anti_aliasing=True, preserve_range=True) for img in
                img_list]
    return img_list


def TensorAugment_2_dim(img_list):
    pil_modes = [x.mode for x in img_list]
    img_list = [np.array(x) for x in img_list]
    img_list = [img.cpu().numpy() for img in img_list] if isinstance(img_list[0], Tensor) else img_list
    if random.random() > 0.5:
        img_list = [np.flip(img, axis=0) for img in img_list]
    if random.random() > 0.5:
        img_list = [np.flip(img, axis=1) for img in img_list]
    if random.random() > 0.5:
        angle = random.random() * 90 - 45
        img_list = [rotate(img, axes=(0, 1), angle=angle, reshape=False, mode='constant', prefilter=True, order=4) for
                    img in img_list]
    if random.random() > 0.5:
        (w, h) = img_list[0].shape
        crop = random.uniform(0.85, 0.95)
        W = int(crop * w)
        H = int(crop * h)
        start_x = w - W
        start_y = h - H
        x_pos = int(random.uniform(0, start_x))
        y_pos = int(random.uniform(0, start_y))
        img_list = [img[y_pos:y_pos + H, x_pos:x_pos + W] for img in img_list]

    img_list = [resize(img, output_shape=(256, 256), anti_aliasing=True, preserve_range=True) for img in
                img_list]
    img_list = [Image.fromarray(x).convert(mode=m) for x, m in zip(img_list, pil_modes)]
    return img_list


def segment_transform(size):
    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(size, interpolation=Image.NEAREST),
        ToLabel()
    ])
    return {'img': img_transform,
            'gt': mask_transform}


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
