from generalframework.models import Segmentator
from generalframework.trainer import Trainer
from generalframework.dataset import MedicalImageDataset, segment_transform, augment, get_dataloaders
from generalframework.loss import get_loss_fn
import yaml, os
import torch
import torch.nn as nn
import warnings

with open('config.yaml', 'r') as f:
    config = yaml.load(f.read())

print('->> Config:\n',config)

dataloders = get_dataloaders(config['Dataset'], config['Dataloader'])

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

trainer = Trainer(model, dataloaders=dataloders, criterion=criterion, **config['Trainer'])
trainer.start_training()
