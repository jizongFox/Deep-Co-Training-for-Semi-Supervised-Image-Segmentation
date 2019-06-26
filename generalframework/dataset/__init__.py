import re
from copy import deepcopy as dcopy
from typing import List, Dict, Callable

import numpy as np
from torch.utils.data import DataLoader

from generalframework.utils import Colorize
from .augment import PILaugment, get_composed_augmentations
from .augment import segment_transform
from .citiyscapesDataloader import CityscapesDataset
from .medicalDataLoader import MedicalImageDataset
from .metainfoGenerator import *

color_transform: Callable = Colorize()

dataset_root: dict = {}


def _registre_data_root(name, root, alis=None):
    if name in dataset_root.keys():
        raise ('The {} has been taken in the dictionary.'.format(name))
    dataset_root[name] = root
    if alis is not None and alis not in dataset_root.keys():
        dataset_root[alis] = root


_registre_data_root('ACDC_2D', './generalframework/dataset/ACDC-2D-All', 'cardiac')
_registre_data_root('PROSTATE', './generalframework/dataset/PROSTATE', 'prostate')


def get_dataset_root(dataname):
    if dataname in dataset_root.keys():
        return dataset_root[dataname]
    else:
        raise ('There is no such dataname, given {}'.format(dataname))


'''
def get_spin_splite_dataset(config):
    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'],mode1='train',mode2='unlabeled')
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
'''
