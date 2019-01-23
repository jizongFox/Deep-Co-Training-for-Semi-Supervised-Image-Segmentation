import warnings
from pprint import pprint
from generalframework.dataset import get_dataloaders, extract_patients
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import VatTrainer
from generalframework.utils import yaml_parser, dict_merge
import yaml

warnings.filterwarnings('ignore')

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config_vat.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

dataloders = get_dataloaders(config['Dataset'], config['Lab_Dataloader'])
lab_dataloader = extract_patients(dataloders['train'], [str(x) for x in range(1, 50)])
unlab_dataloader = get_dataloaders(config['Dataset'], config['Unlab_Dataloader'], quite=True)['train']
unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(50, 100)])
dataloders = {'lab': lab_dataloader,
              'unlab': unlab_dataloader,
              'val': dataloders['val']}

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

vattrainer = VatTrainer(model, dataloaders=dataloders, criterion=criterion, **config['Trainer'], whole_config=config)
vattrainer.start_training(**config['StartTraining'])
