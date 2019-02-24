import os
import random
import warnings
from pprint import pprint

import numpy as np
import torch
import yaml

from generalframework.dataset import get_ACDC_split_dataloders
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



labeled_dataloaders, unlab_dataloader, val_dataloader = get_ACDC_split_dataloders(config)

def get_models(config):
    num_models = config['Lab_Partitions']['num_models']
    for i in range(num_models):
        return [Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
                for _ in range(num_models)]

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
