import warnings
from pprint import pprint

import numpy as np
import os
import random
import torch
import yaml

from generalframework.dataset import get_ACDC_dataloaders, extract_patients
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import VatTrainer
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


def get_dataloders(config):
    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'])
    labeled_dataloaders = []
    for i in config['Lab_Partitions']['label']:
        labeled_dataloaders.append(extract_patients(dataloders['train'], [str(x) for x in range(*i)]))

    unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
    unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(*config['Lab_Partitions']['unlabel'])])
    val_dataloader = dataloders['val']
    return labeled_dataloaders, unlab_dataloader, val_dataloader


fix_seed(1234)

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config/config_vat.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

labeled_dataloaders, unlab_dataloader, val_dataloader = get_dataloders(config)
dataloaders = {'lab': labeled_dataloaders[0],
               'unlab': unlab_dataloader,
               'val': val_dataloader}

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

vattrainer = VatTrainer(segmentator=model,
                        dataloaders=dataloaders,
                        criterion=criterion,
                        **config['Trainer'],
                        whole_config=config,
                        adv_scheduler_dict=config['Adv_Scheduler']
                        )
vattrainer.start_training(**config['StartTraining'], adv_training_dict=config['Adv_Training'])
