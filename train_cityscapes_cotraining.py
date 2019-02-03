import warnings
from pprint import pprint

import numpy as np
import os
import random
import torch
import yaml

from generalframework.dataset import get_dataloaders, extract_cities, get_cityscapes_dataloaders, citylist
from generalframework.loss import get_loss_fn,enet_weighing
from generalframework.models import Segmentator
from generalframework.trainer import CoTrainer_City
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
with open('cityscapes_config_cotrain.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

dataloders = get_cityscapes_dataloaders(config['Dataset'], config['Lab_Dataloader'])
lab_dataloader1 = extract_cities(dataloders['train'],
                                 [city for city in citylist if city not in ['stuttgart', 'ulm', 'zurich']])
lab_dataloader2 = extract_cities(dataloders['train'],
                                 [city for city in citylist if city not in ['aachen', 'bremen', 'darmstadt']])
unlab_dataloader = get_cityscapes_dataloaders(config['Dataset'], config['Unlab_Dataloader'])['train']

val_dataloader = dataloders['val']

model1 = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
model2 = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
# model2.load_state_dict(model1.state_dict)
# class_weights = enet_weighing(dataloders['train'],19)
# print(class_weights)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterions = {'sup': get_loss_fn('cross_entropy',**{'weight':config['Loss']['weight'],'ignore_index' : 255}),
                  'jsd': get_loss_fn('jsd'),
                  'adv': get_loss_fn('jsd')}

cotrainner = CoTrainer_City(segmentators=[model1, model2],
                            labeled_dataloaders=[lab_dataloader1, lab_dataloader2],
                            unlabeled_dataloader=unlab_dataloader,
                            val_dataloader=val_dataloader,
                            criterions=criterions,
                            **config['Trainer'],
                            whole_config=config)

cotrainner.start_training(**config['StartTraining'])
