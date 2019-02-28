# coding=utf8
from __future__ import print_function, division
import os, sys, random, re
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from generalframework import ModelMode
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional, TypeVar, Iterable
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
import torch, numpy as np
from .metainfoGenerator import *
from .augment import PILaugment, segment_transform, temporary_seed


class MedicalImageDataset(Dataset):
    dataset_modes = ['train', 'val', 'test','unlabeled']
    allow_extension = ['.jpg', '.png']

    def __init__(self, root_dir: str, mode: str, subfolders: List[str], transform=None, augment=None,
                 equalize: Union[List[str], str, None] = None,
                 pin_memory=False, metainfo: str = None, quite=False):
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
            if isinstance(metainfo[0],str):
                metainfo[0]=eval(metainfo[0])
                metainfo[1]=eval(metainfo[1]) if isinstance(metainfo[1],str) else metainfo[1]
            self.metainfo_generator = metainfo[0](**metainfo[1])

    def __len__(self):
        return int(len(self.imgs[self.subfolders[0]]))

    def set_mode(self, mode):
        assert isinstance(mode, (str, ModelMode)), 'the type of mode should be str or ModelMode, given %s' % str(mode)

        if isinstance(mode, str):
            self.training = ModelMode.from_str(mode)
        else:
            self.training = mode

    def __getitem__(self, index):
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
            random_seed = (random.getstate(),np.random.get_state())
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
        return img_T, [metainformation,str(random_seed)] if 'random_seed' in locals() else [metainformation,Tensor([1])], filename

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
        ## clear up extension
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


def id_(x):
    return x


A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", torch.Tensor, np.ndarray)


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageDataset, grp_regex, shuffle=False, quite=False) -> None:
        filenames: List[str] = dataset.filenames[dataset.subfolders[0]]
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_
        if not quite:
            print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(1) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(filenames)
        if not quite:
            print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)
        if not quite:
            print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)
