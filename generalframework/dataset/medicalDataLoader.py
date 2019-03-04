# coding=utf8
from __future__ import print_function, division
import os, random, re
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from generalframework import ModelMode
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional, TypeVar, Iterable
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from .metainfoGenerator import *
from .augment import PILaugment, segment_transform, temporary_seed
from ..utils import map_


class MedicalImageDataset(Dataset):
    dataset_modes = ['train', 'val', 'test', 'unlabeled']
    allow_extension = ['.jpg', '.png']

    def __init__(self, root_dir: str, mode: str, subfolders: List[str], transform=None, augment=None,
                 equalize: Union[List[str], str, None] = None,
                 pin_memory=False, metainfo: str = None, quite=False) -> None:
        '''
        :param root_dir: dataset root
        :param mode: train or test or val etc, should be in the cls attribute
        :param subfolders: subfolder names in the mode folder
        :param transform: image trans        print(imgs.shape)formation
        :param augment: image and gt augmentation
        :param equalize:
        '''
        assert len(subfolders) == set(subfolders).__len__()
        self.name = '%s_dataset' % mode
        self.mode = mode
        self.root_dir = root_dir
        self.subfolders = subfolders
        self.transform = eval(transform) if isinstance(transform, str) else transform
        self.pin_memory = pin_memory
        if not quite:
            print(f'->> Building {self.name}:\t')
        self.imgs, self.filenames = self.make_dataset(self.root_dir, self.mode, self.subfolders, self.pin_memory,
                                                      quite=quite)
        self.augment = eval(augment) if isinstance(augment, str) else augment
        self.equalize = equalize
        self.training = ModelMode.TRAIN
        if metainfo is None:
            self.metainfo_generator = None
        else:
            if isinstance(metainfo[0], str):
                metainfo[0]: Callable = eval(metainfo[0])
                metainfo[1]: dict = eval(metainfo[1]) if isinstance(metainfo[1], str) else metainfo[1]
            self.metainfo_generator: Callable = metainfo[0](**metainfo[1])

    def __len__(self) -> int:
        return int(len(self.imgs[self.subfolders[0]]))

    def set_mode(self, mode) -> None:
        assert isinstance(mode, (str, ModelMode)), 'the type of mode should be str or ModelMode, given %s' % str(mode)

        if isinstance(mode, str):
            self.training = ModelMode.from_str(mode)
        else:
            self.training = mode

    def __getitem__(self, index) -> Tuple[list, Union[List[Union[list, Tensor, str]], List[Union[list, Tensor]]], str]:
        if self.pin_memory:
            img_list, filename_list = [self.imgs[subfolder][index] for subfolder in self.subfolders], [
                self.filenames[subfolder][index] for subfolder in self.subfolders]
        else:
            img_list, filename_list = [Image.open(self.imgs[subfolder][index]) for subfolder in self.subfolders], [
                self.filenames[subfolder][index] for subfolder in self.subfolders]
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        filename = Path(filename_list[0]).stem

        if self.equalize:
            img_list = [ImageOps.equalize(img) if (b == self.equalize) or (b in self.equalize) else img for b, img in
                        zip(self.subfolders, img_list)]

        if self.augment is not None and self.training == ModelMode.TRAIN:
            random_seed = (random.getstate(), np.random.get_state())
            A_img_list = self.augment(img_list)

            img_T = [self.transform['img'](img) if b == 'img' else self.transform['gt'](img) for b, img in
                     zip(self.subfolders, A_img_list)]
        else:
            img_T = [self.transform['img'](img) if b == 'img' else self.transform['gt'](img) for b, img in
                     zip(self.subfolders, img_list)]

        metainformation = torch.Tensor([-1])
        if self.metainfo_generator is not None:
            original_imgs = [self.transform['img'](img) if b == 'img' else self.transform['gt'](img) for b, img in
                             zip(self.subfolders, img_list)]
            metainformation = [self.metainfo_generator(img_t) for b, img_t in
                               zip(self.subfolders, original_imgs) if b in self.metainfo_generator.foldernames]
        # random_seed=1
        return img_T, [metainformation, str(random_seed)] if 'random_seed' in locals() else [metainformation,
                                                                                             Tensor([1])], filename

    @classmethod
    def make_dataset(cls, root, mode, subfolders, pin_memory, quite=False):
        def allow_extension(path: str, extensions: List[str]) -> bool:
            try:
                return Path(path).suffixes[0] in extensions
            except:
                return False

        assert mode in cls.dataset_modes

        for subfolder in subfolders:
            assert Path(os.path.join(root, mode, subfolder)).exists(), Path(os.path.join(root, mode, subfolder))

        items = [os.listdir(Path(os.path.join(root, mode, subfoloder))) for subfoloder in
                 subfolders]
        # clear up extension
        items = [[x for x in item if allow_extension(x, cls.allow_extension)] for item in items]
        assert set(map_(len, items)).__len__() == 1, map_(len, items)

        imgs = {}

        for subfolder, item in zip(subfolders, items):
            imgs[subfolder] = [os.path.join(root, mode, subfolder, x_path) for x_path in item]

        assert set(map_(len, imgs.values())).__len__() == 1

        for subfolder in subfolders:
            if not quite:
                print(f'found {len(imgs[subfolder])} images in {subfolder}\t')

        if pin_memory:
            if not quite:
                print(f'pin_memory in progress....')
            pin_imgs = {}
            for k, v in imgs.items():
                pin_imgs[k] = [Image.open(i).convert('L') for i in v]
            if not quite:
                print(f'pin_memory sucessfully..')
            return pin_imgs, imgs

        return imgs, imgs
