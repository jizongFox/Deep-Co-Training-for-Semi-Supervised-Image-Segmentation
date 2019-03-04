from torch.utils.data import DataLoader
from . import MedicalImageDataset


def get_GM_dataloaders(dataset_dict: dict, dataloader_dict: dict, quite=False, mode1='train', mode2='val'):
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
