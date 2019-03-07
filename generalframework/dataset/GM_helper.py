from torch.utils.data import DataLoader
from typing import List, Union, Dict
import re
from copy import deepcopy as dcopy
from . import MedicalImageDataset
from .augment import segment_transform, PILaugment
import numpy as np
from pathlib import Path
from functools import reduce


# todo make it work for GM dataset

def get_GM_dataloaders(dataset_dict: dict, dataloader_dict: dict, quite=False, mode1='train', mode2='unlabeled') -> \
        Dict[str, DataLoader]:
    '''
    return the train and unlabeled datasets as defined in the MedicalImageDataset
    :param dataset_dict:
    :param dataloader_dict:
    :param quite: True
    :param mode1: train
    :param mode2: unlabeled
    :return:
    '''
    dataset_dict = {k: eval(v) if isinstance(v, str) and k != 'root_dir' else v for k, v in dataset_dict.items()}
    dataloader_dict = {k: eval(v) if isinstance(v, str) else v for k, v in dataloader_dict.items()}
    train_set = MedicalImageDataset(mode=mode1, quite=quite, **dataset_dict)
    unlabeled_set = MedicalImageDataset(mode=mode2, quite=quite, **dataset_dict)
    train_loader = DataLoader(train_set, **{**dataloader_dict, **{'batch_sampler': None}})
    val_loader = DataLoader(unlabeled_set, **{**dataloader_dict, **{'batch_sampler': None}})
    return {'train': train_loader, 'unlabeled': val_loader}


def get_GMC_split_dataloders(config):
    def assert_same_dataset(dataloader: DataLoader):
        assert set(map(lambda x: Path(x).name, dataloader.dataset.filenames['img'])) == set(
            map(lambda x: Path(x).name, dataloader.dataset.filenames['gt']))

    def no_overlap_dataloader(dataloader1: DataLoader, dataloader2: DataLoader):
        assert_same_dataset(dataloader1)
        assert_same_dataset(dataloader2)
        assert len(set(map(lambda x: Path(x).name, dataloader1.dataset.filenames['img'])) & set(
            map(lambda x: Path(x).name, dataloader2.dataset.filenames['img']))) == 0
        return dataloader2

    gm_dataloader = get_GM_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)
    lab_dataloader, unlabeled_dataloader = gm_dataloader['train'], gm_dataloader['unlabeled']
    val_dataloader = extract_patients_gmc(lab_dataloader, site_id=[1, 2])
    train_dataloader = extract_patients_gmc(lab_dataloader, site_id=[3, 4])
    print('-> Building dataloaders:\n'
          'train_image_num:', train_dataloader.dataset.__len__(),
          ' unlab_image_num:', unlabeled_dataloader.dataset.__len__(),
          ' val_img_num:', val_dataloader.dataset.__len__())
    # assert no overlap between them.
    reduce(lambda x, y: no_overlap_dataloader(x, y), [lab_dataloader, unlabeled_dataloader, val_dataloader])
    u_pattern = re.compile(r'site\d-sc\d\d')
    u_samples = sorted(list(set([u_pattern.findall(x)[0] for x in train_dataloader.dataset.filenames['img']])))

    num_model = int(config["Lab_Partitions"]["num_models"])
    print(f'Found {u_samples.__len__()} unique experiments in lab_dataloader, try to split them to {num_model} models'
          f'with overlap = {config["Lab_Partitions"]["partition_overlap"]}')

    common_patterns = list(
        np.random.choice(u_samples, int(len(u_samples) * float(config["Lab_Partitions"]["partition_overlap"])),
                         replace=False))
    exclusive_patterns = [x for x in u_samples if x not in common_patterns]
    assert (set(common_patterns) & set(exclusive_patterns)).__len__() == 0
    pattern_per_loader = [list(common_patterns) + exclusive_patterns[i::num_model] for i in range(num_model)]

    labeled_dataloaders = []
    for i in range(num_model):
        labeled_dataloaders.append(extract_patients_gmc(dataloader=train_dataloader, pattern=pattern_per_loader[i]))

    print(f'{len(labeled_dataloaders)} datasets with overlap labeled patients',
          len(reduce(lambda x, y: x & y, list(map(lambda x: set(x), pattern_per_loader)))))

    return labeled_dataloaders, unlabeled_dataloader, val_dataloader


def extract_patients_gmc(dataloader: DataLoader, site_id: Union[List[int], None] = [1, 2], pattern=None) -> DataLoader:
    if pattern is not None:
        patterns = re.compile('|'.join(
            pattern
        ))
    else:
        assert isinstance(site_id, list)

        bpattern = lambda site: 'site{:01d}'.format(int(site))
        patterns = re.compile('|'.join(
            [bpattern(site) for site in site_id]
        ))

    # patterns = re.compile('|'.join([bpattern(id) for id in patient_ids]))
    files: Dict[str, List[str]] = dcopy(dataloader.dataset.imgs)
    files = {k: [s for s in file if re.search(patterns, s)] for k, file in files.items()}
    for v in files.values():
        v.sort()
    new_dataloader = dcopy(dataloader)
    new_dataloader.dataset.imgs = files
    new_dataloader.dataset.filenames = files
    return new_dataloader
