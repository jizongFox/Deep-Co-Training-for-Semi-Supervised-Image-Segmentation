import random
import numpy as np
from copy import deepcopy as dcopy
from torch.utils.data import DataLoader
from PIL import ImageOps
from torchvision import transforms
from .augment import PILaugment
from .augment import segment_transform

from generalframework.utils import Colorize
from .medicalDataLoader import MedicalImageDataset, PatientSampler

color_transform = Colorize()

dataset_root = {}


def _registre_data_root(name, root, alis=None):
    if name in dataset_root.keys():
        raise ('The {} has been taken in the dictionary.'.format(name))
    dataset_root[name] = root
    if alis is not None and alis not in dataset_root.keys():
        dataset_root[alis] = root


_registre_data_root('ACDC_2D', 'generalframework/dataset/ACDC-2D-All', 'cardiac')
_registre_data_root('PROSTATE', 'generalframework/dataset/PROSTATE', 'prostate')


def get_dataset_root(dataname):
    if dataname in dataset_root.keys():
        return dataset_root[dataname]
    else:
        raise ('There is no such dataname, given {}'.format(dataname))


def get_dataloaders(dataset_dict: dict, dataloader_dict: dict):
    dataset_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataset_dict.items()}
    dataloader_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataloader_dict.items()}
    train_set = MedicalImageDataset(mode='train', **dataset_dict)
    val_set = MedicalImageDataset(mode='val', **dataset_dict)
    train_loader = DataLoader(train_set, **{**dataloader_dict, **{'batch_sampler': None}})

    if dataloader_dict.get('batch_sampler') is not None:

        val_sampler = eval(dataloader_dict.get('batch_sampler')[0])(dataset=val_set,
                                                                    **dataloader_dict.get('batch_sampler')[1])
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, batch_size=1)
    else:
        val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'train': train_loader,
            'val': val_loader}


def get_exclusive_dataloaders(dataset_dict: dict, dataloader_dict:dict, n_models=2, ratio=0.5):
    # validate split ratio for labeled and unlabeled
    assert np.array(ratio).min() > 0 and np.array(ratio).max() < 1

    dataset_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataset_dict.items()}
    dataloader_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataloader_dict.items()}
    lab_dataset = MedicalImageDataset(mode='train', **dataset_dict)
    unlab_dataset = dcopy(lab_dataset)
    lab_datasets = [dcopy(lab_dataset) for _ in range(n_models)]

    # split train_set in labeled and unlabeled data
    np.random.seed(1)
    data = list(zip(lab_dataset.filenames['img'], lab_dataset.filenames['gt']))
    idx = np.random.choice(len(data), int(ratio * data.__len__()), replace=False)
    unlabeled_data = [data[i] for i in range(len(data)) if i not in idx]

    imgs, gts = zip(*unlabeled_data)
    unlab_dataset.filenames['img'] = imgs
    unlab_dataset.filenames['gt'] = gts

    n_labimgs_per_set = int((lab_dataset.__len__() - unlabeled_data.__len__()) / n_models)
    labeled_data = [data[i] for i in range(len(data)) if i in idx]

    # creating the 2 exclusive labeled sets
    imgs, gts = zip(*labeled_data)
    for i in range(n_models):
        lab_datasets[i].filenames['img'] = imgs[i*n_labimgs_per_set:(i+1)*n_labimgs_per_set]
        lab_datasets[i].filenames['gt'] = gts[i*n_labimgs_per_set:(i+1)*n_labimgs_per_set]

    lab_loaders = [DataLoader(x, **{**dataloader_dict, **{'batch_sampler': None}}) for x in lab_datasets]
    unlab_loader = DataLoader(unlab_dataset, **{**dataloader_dict, **{'batch_sampler': None}})

    val_dataset = MedicalImageDataset(mode='val', **dataset_dict)
    if dataloader_dict.get('batch_sampler') is not None:

        val_sampler = eval(dataloader_dict.get('batch_sampler')[0])(dataset=val_dataset,
                                                                    **dataloader_dict.get('batch_sampler')[1])
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, batch_size=1)
    else:
        val_loader = DataLoader(val_dataset, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'lab': lab_loaders,
            'unlab': unlab_loader,
            'val': val_loader}
