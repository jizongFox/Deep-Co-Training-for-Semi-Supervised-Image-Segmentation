import random
from pathlib import Path
from itertools import repeat
import re

import torch
import numpy as np
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional, TypeVar, Iterable
from torch.utils.data import DataLoader, Sampler
from copy import deepcopy as dcopy
from . import MedicalImageDataset
from .augment import segment_transform, PILaugment,TensorAugment_2_dim
from ..utils import export

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", torch.Tensor, np.ndarray)


def id_(x):
    return x


def map_(fn: Callable[[A], B], iter_: Iterable[A]) -> List[B]:
    return list(map(fn, iter_))


@export
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


@export
def get_ACDC_dataloaders(dataset_dict: dict, dataloader_dict: dict, quite=False, mode1='train', mode2='val'):
    dataset_dict = {k: eval(v) if isinstance(v, str) and k != 'root_dir' else v for k, v in dataset_dict.items()}
    dataloader_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataloader_dict.items()}
    train_set = MedicalImageDataset(mode=mode1, quite=quite, **dataset_dict)
    val_set = MedicalImageDataset(mode=mode2, quite=quite, **dataset_dict)
    train_loader = DataLoader(train_set, **{**dataloader_dict, **{'batch_sampler': None}})

    if dataloader_dict.get('batch_sampler') is not None:
        val_sampler = eval(dataloader_dict.get('batch_sampler')[0]) \
            (dataset=val_set, **dataloader_dict.get('batch_sampler')[1])
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, batch_size=1)
    else:
        val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'train': train_loader, 'val': val_loader}


@export
def get_ACDC_split_dataloders(config):
    def create_partitions(partition_ratio):
        lab_ids = [1, int(100 * partition_ratio + 1)]
        unlab_ids = [int(100 * partition_ratio + 1), 101]
        return lab_ids, unlab_ids

    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'])
    partition_ratio = config['Lab_Partitions']['partition_sets']
    lab_ids, unlab_ids = create_partitions(partition_ratio)
    partition_overlap = config['Lab_Partitions']['partition_overlap']
    rd_idx = np.random.permutation(range(*lab_ids))
    overlap_idx = np.random.choice(rd_idx, size=int(float(partition_overlap) * range(*lab_ids).__len__()),
                                   replace=False)
    exclusive_idx = [x for x in rd_idx if x not in overlap_idx]
    n_splits = int(config['Lab_Partitions']['num_models'])
    exclusive_samples = int(exclusive_idx.__len__() / n_splits)
    excl_indx = [exclusive_idx[i * exclusive_samples: (i + 1) * exclusive_samples] for i in range(n_splits)]

    lab_partitions = [np.hstack((overlap_idx, np.array(excl_indx[idx]))) for idx in range(n_splits)]
    labeled_dataloaders = []
    for idx_lst in lab_partitions:
        labeled_dataloaders.append(extract_patients(dataloders['train'], [str(int(x)) for x in idx_lst]))

    unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
    unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(*unlab_ids)])
    val_dataloader = dataloders['val']
    print('labeled_image_number:', len(range(*lab_ids)), 'unlab_image_number:', len(range(*unlab_ids)))
    from functools import reduce
    print(f'{len(lab_partitions)} datasets with overlap labeled image number',
          len(reduce(lambda x, y: x & y, list(map(lambda x: set(x), lab_partitions)))))
    return labeled_dataloaders, unlab_dataloader, val_dataloader


def extract_patients(dataloader: DataLoader, patient_ids: List[str]):
    '''
     extract patients from ACDC dataset provding patient_ids
    :param dataloader:
    :param patient_ids:
    :return:
    '''
    assert isinstance(patient_ids, list)
    bpattern = lambda d: 'patient%.3d' % int(d)
    patterns = re.compile('|'.join([bpattern(id) for id in patient_ids]))
    files: Dict[str, List[str]] = dcopy(dataloader.dataset.imgs)
    files = {k: [s for s in file if re.search(patterns, s)] for k, file in files.items()}
    for v in files.values():
        v.sort()
    new_dataloader = dcopy(dataloader)
    new_dataloader.dataset.imgs = files
    new_dataloader.dataset.filenames = files
    return new_dataloader
