import os
from pathlib import Path

import numpy as np
import scipy.misc as m
import torch
from generalframework import ModelMode
from torch.utils.data import Dataset

from ..utils.utils import recursive_glob


class CityscapesDataset(Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py

    Dataloader for Cityscapes dataset (https://www.cityscapes-dataset.com).
    The data_utils packages gtFine_trainvaltest and leftImg8bit_trainvaltest can be downloaded from:
    https://www.cityscapes-dataset.com/downloads/

    Thanks a lot to @meetshah1995 for the loader repo:
   https://github.com/meetshah1995/pytorch-semseg/tree/master/ptsemseg/loader/cityscapes_loader.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    CITYSCAPES_MEAN = [0.290101, 0.328081, 0.286964]
    CITYSCAPES_STD = [0.182954, 0.186566, 0.184475]

    ## RGB channels

    def __init__(self, root_path: str, mode: str = "train", is_transform: bool = False,
                 augmentation=None, image_size=(768, 1024), quite: bool = False):
        """__init__
        :param root_path:
        :param mode:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = Path(root_path)
        assert self.root.exists(), self.root
        self.mode = mode
        assert mode in ('train', 'test', 'val'), 'dataset name should be restricted in "label", \
        "unlabeled" and "val", given %s' % mode
        self.is_transform = is_transform
        self.augmentations = augmentation
        self.training = ModelMode.TRAIN
        self.num_classes = 19
        self.files: dict = {}
        self.img_size = eval(image_size) if type(image_size) == str else image_size

        self.images_base = self.root / "leftImg8bit" / self.mode
        self.annotations_base = self.root / "gtFine" / self.mode
        assert self.images_base.exists(), self.images_base
        assert self.annotations_base.exists(), self.annotations_base

        self.files[mode]: list = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ["unlabelled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                            "traffic_sign",
                            "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                            "motorcycle",
                            "bicycle", ]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if self.files[mode].__len__() == 0:
            raise Exception(
                "No files for split=[%s] found in %s" % (mode, self.images_base)
            )
        if not quite:
            print("Found %d %s images" % (len(self.files[mode]), mode))

    def __len__(self):
        """__len__"""
        return int(len(self.files[self.mode]))

    def set_mode(self, mode):
        assert isinstance(mode, (str, ModelMode)), 'the type of mode should be str or ModelMode, given %s' % str(mode)

        if isinstance(mode, str):
            self.training = ModelMode.from_str(mode)
        else:
            self.training = mode

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.mode][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)

        img = np.array(img, dtype=np.uint8)
        img = m.imresize(
            img, (int(self.img_size[0]), int(self.img_size[1]))
        )  # uint8 with RGB mode

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")

        if self.augmentations is not None and self.training == ModelMode.TRAIN:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        lbl = lbl.unsqueeze(0)

        return [img, lbl], torch.ones(1), lbl_path

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """

        img = img.astype(np.float64) / 255.0
        img -= self.CITYSCAPES_MEAN
        img /= self.CITYSCAPES_STD

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)

        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            # print("WARN: resizing labels yielded fewer classes")
            pass
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.num_classes):
            # print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        if len(temp.shape) == 3:
            temp = temp.squeeze()
        assert len(temp.shape) == 2
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.num_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
