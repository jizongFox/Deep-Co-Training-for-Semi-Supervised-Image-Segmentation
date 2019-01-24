import re
import numpy as np
from copy import deepcopy as dcopy
from typing import List, Dict
from copy import deepcopy as dcopy
from torch.utils.data import DataLoader
from generalframework.utils import Colorize
from .augment import PILaugment, get_composed_augmentations
from .augment import segment_transform
from .medicalDataLoader import MedicalImageDataset, PatientSampler
from .citiyscapesDataloader import CityscapesDataset

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
_registre_data_root('CITYSCAPES', 'generalframework/dataset/CITYSCAPES', 'cityscapes')


def get_dataset_root(dataname):
    if dataname in dataset_root.keys():
        return dataset_root[dataname]
    else:
        raise ('There is no such dataname, given {}'.format(dataname))


# TODO: Implement the get_dataloaders for Cityscapes dataset in a similar way as MedicalImageDataset
def get_cityscapes_dataloaders(dataset_dict: dict, dataloader_dict: dict, n_models=2, ratio=0.5):
    # Setup Augmentations
    augmentations = dataset_dict.get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_path = dataset_dict["root_dir"]

    lab_set = CityscapesDataset(
        data_path,
        is_transform=True,
        mode='train',
        img_size=(dataset_dict["img_rows"], dataset_dict["img_cols"]),
        augmentation=data_aug,
    )

    unlab_set = dcopy(lab_set)
    lab_datasets = [dcopy(lab_set) for _ in range(n_models)]
    lab_set_len = len(lab_set.files['train'])

    np.random.seed(1)
    idx = np.random.choice(lab_set_len, int(ratio * lab_set_len), replace=False)
    unlab_data = [lab_set.files['train'][i] for i in range(lab_set_len) if i not in idx]
    unlab_set.files['train'] = unlab_data

    n_labimgs_per_set = int((lab_set_len - unlab_data.__len__()) / n_models)
    lab_data = [lab_set.files['train'][i] for i in range(lab_set_len) if i in idx]

    for i in range(n_models):
        lab_datasets[i].files['train'] = lab_data[i*n_labimgs_per_set:(i+1)*n_labimgs_per_set]

    lab_loaders = [DataLoader(x, **dataloader_dict) for x in lab_datasets]
    unlab_loader = DataLoader(unlab_set,  **dataloader_dict)

    val_set = CityscapesDataset(
        data_path,
        is_transform=True,
        mode='val',
        img_size=(dataset_dict["img_rows"], dataset_dict["img_cols"]),
    )
    val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {
        'lab': lab_loaders,
        'unlab': unlab_loader,
        'val': val_loader}


def get_dataloaders(dataset_dict: dict, dataloader_dict: dict, quite=False):
    dataset_dict = {k: eval(v) if isinstance(v, str) and k != 'root_dir' else v for k, v in dataset_dict.items()}
    dataloader_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataloader_dict.items()}
    train_set = MedicalImageDataset(mode='train', **dataset_dict, quite=quite)
    val_set = MedicalImageDataset(mode='val', **dataset_dict, quite=quite)
    train_loader = DataLoader(train_set, **{**dataloader_dict, **{'batch_sampler': None}})

    if dataloader_dict.get('batch_sampler') is not None:

        val_sampler = eval(dataloader_dict.get('batch_sampler')[0])(dataset=val_set,
                                                                    **dataloader_dict.get('batch_sampler')[1])
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, batch_size=1)
    else:
        val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'train': train_loader,
            'val': val_loader}


def extract_patients(dataloader: DataLoader, patient_ids: List[str]):
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
