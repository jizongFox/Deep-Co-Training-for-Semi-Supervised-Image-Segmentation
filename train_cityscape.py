import warnings
from pprint import pprint
from generalframework.dataset import get_dataloaders, get_cityscapes_dataloaders
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import Trainer_City
from generalframework.utils import yaml_parser, dict_merge
import yaml

warnings.filterwarnings('ignore')

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('tmp.yml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

dataloders = get_cityscapes_dataloaders(config['Dataset'], config['Dataloader'])

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

trainer = Trainer_City(model, dataloaders=dataloders, criterion=criterion, **config['Trainer'], whole_config=config)
trainer.start_training(**config['StartTraining'])
