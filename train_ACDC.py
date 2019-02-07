import warnings
from pprint import pprint

import numpy as np
import os
import random
import torch
import yaml

from generalframework.dataset import get_ACDC_dataloaders
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import Trainer
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
with open('config/ACDC_config.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

dataloders = get_ACDC_dataloaders(config['Dataset'], config['Dataloader'])

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

trainer = Trainer(segmentator=model,
                  dataloaders=dataloders,
                  criterion=criterion,
                  **config['Trainer'],
                  whole_config=config)
trainer.start_training(**config['StartTraining'])
