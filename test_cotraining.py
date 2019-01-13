from generalframework.models import Segmentator
from generalframework.trainer import Trainer, CoTrainer
from generalframework.dataset import MedicalImageDataset, segment_transform, augment, get_dataloaders
from generalframework.loss import get_loss_fn
import yaml, os
from copy import deepcopy as dcopy
import torch
import torch.nn as nn
import warnings

with open('config_cotrain.yaml', 'r') as f:
    config = yaml.load(f.read())

print('->> Config:\n', config)

dataloders = get_dataloaders(config['Dataset'], config['Dataloader'])

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=UserWarning)
#     criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterions = {'sup': get_loss_fn('cross_entropy'),
                  'jsd': get_loss_fn('jsd'),
                  'adv': get_loss_fn('jsd')}

cotrainner = CoTrainer(segmentators=[model, dcopy(model)],
                       labeled_dataloaders=[dataloders['train'], dcopy(dataloders['train'])],
                       unlabeled_dataloader=dcopy(dataloders['train']),
                       val_dataloader=dataloders['val'],
                       criterions=criterions,
                       **config['Trainer'])
cotrainner.start_training()
