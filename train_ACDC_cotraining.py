import os
import random
import warnings
from pprint import pprint

import numpy as np
import torch
import yaml

from generalframework.dataset import get_ACDC_dataloaders, extract_patients
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import CoTrainer
from generalframework.utils import yaml_parser, dict_merge

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config/ACDC_config_cotrain.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

fix_seed(int(config['Seed']))

def get_models(config):
    num_models = config['Lab_Partitions']['label'].__len__()
    for i in range(num_models):
        return [Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
                for _ in range(num_models)]


def get_dataloders(config):
    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'])
    try:
        partition_ratio = config['Lab_Partitions']['partition_sets']
        lab_ids, unlab_ids = create_partitions(partition_ratio)
        partition_overlap = config['Lab_Partitions']['partition_overlap']
        rd_idx = np.random.permutation(range(*lab_ids))
        overlap_idx = np.random.choice(rd_idx, size=int(partition_overlap * range(*lab_ids).__len__()),
                                       replace=False)
        exclusive_idx = [x for x in rd_idx if x not in overlap_idx]
        n_splits = config['Lab_Partitions']['label'].__len__()
        exclusive_samples = int(exclusive_idx.__len__() / n_splits)
        excl_indx = [exclusive_idx[i * exclusive_samples: (i + 1) * exclusive_samples] for i in range(n_splits)]

        lab_partitions = [np.hstack((overlap_idx, np.array(excl_indx[idx]))) for idx in range(n_splits)]
        labeled_dataloaders = []
        for idx_lst in lab_partitions:
            labeled_dataloaders.append(extract_patients(dataloders['train'], [str(int(x)) for x in idx_lst]))

        unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
        unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(*unlab_ids)])
        val_dataloader = dataloders['val']
        print('labeled_image_number:',len(range(*lab_ids)), 'unlab_image_number:',len(range(*unlab_ids)))
        from functools import reduce
        print(f'{len(lab_partitions)} datasets with overlap labeled image number', len(reduce(lambda x,y: x&y,list(map(lambda x:set(x), lab_partitions)))))
        return labeled_dataloaders, unlab_dataloader, val_dataloader
    except:
        labeled_dataloaders = []
        for i in config['Lab_Partitions']['label']:
            labeled_dataloaders.append(extract_patients(dataloders['train'], [str(x) for x in range(*i)]))

        unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
        unlab_dataloader = extract_patients(unlab_dataloader,
                                            [str(x) for x in range(*config['Lab_Partitions']['unlabel'])])
        val_dataloader = dataloders['val']
        return labeled_dataloaders, unlab_dataloader, val_dataloader


def create_partitions(partition_ratio=0.6):
    lab_ids = [1,int(100* partition_ratio+1)]
    unlab_ids = [int(100*partition_ratio+1), 101]
    return lab_ids, unlab_ids


labeled_dataloaders, unlab_dataloader, val_dataloader = get_dataloders(config)


segmentators = get_models(config)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterions = {'sup': get_loss_fn('cross_entropy'),
                  'jsd': get_loss_fn('jsd'),
                  'adv': get_loss_fn('jsd')}

cotrainner = CoTrainer(segmentators=segmentators,
                       labeled_dataloaders=labeled_dataloaders,
                       unlabeled_dataloader=unlab_dataloader,
                       val_dataloader=val_dataloader,
                       criterions=criterions,
                       adv_scheduler_dict=config['Adv_Scheduler'],
                       cot_scheduler_dict=config['Cot_Scheduler'],
                       adv_training_dict=config['Adv_Training'],
                       **config['Trainer'],
                       whole_config=config)

cotrainner.start_training(**config['StartTraining'])
