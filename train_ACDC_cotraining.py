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

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config/ACDC_config_cotrain.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)


def get_models(config):
    num_models = config['Lab_Partitions']['label'].__len__()
    for i in range(num_models):
        return [Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
                for _ in range(num_models)]


def get_dataloders(config):
    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'])
    labeled_dataloaders = []
    for i in config['Lab_Partitions']['label']:
        labeled_dataloaders.append(extract_patients(dataloders['train'], [str(x) for x in range(*i)]))

    unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
    unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(*config['Lab_Partitions']['unlabel'])])
    val_dataloader = dataloders['val']
    return labeled_dataloaders, unlab_dataloader, val_dataloader


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
                       **config['Trainer'],
                       whole_config=config)

cotrainner.start_training(**config['StartTraining'])
