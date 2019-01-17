import warnings
from pprint import pprint
from generalframework.dataset import get_dataloaders, extract_patients
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import CoTrainer
import yaml

with open('config_cotrain.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Config:')
pprint(config)

dataloders = get_dataloaders(config['Dataset'], config['Lab_Dataloader'])
lab_dataloader1 = extract_patients(dataloders['train'], [str(x) for x in range(1, 26)])
lab_dataloader2 = extract_patients(dataloders['train'], [str(x) for x in range(26, 50)])
unlab_dataloader = get_dataloaders(config['Dataset'], config['Unlab_Dataloader'])['train']
unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(50, 100)])
val_dataloader = dataloders['val']

model1 = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
model2 = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterions = {'sup': get_loss_fn('cross_entropy'),
                  'jsd': get_loss_fn('jsd'),
                  'adv': get_loss_fn('jsd')}

cotrainner = CoTrainer(segmentators=[model1, model2],
                       labeled_dataloaders=[lab_dataloader1, lab_dataloader2],
                       unlabeled_dataloader=unlab_dataloader,
                       val_dataloader=val_dataloader,
                       criterions=criterions,
                       **config['Trainer'])
#
# cotrainner = CoTrainer(segmentators=[model1],
#                        labeled_dataloaders=[dataloders['train']],
#                        unlabeled_dataloader=unlab_dataloader,
#                        val_dataloader=val_dataloader,
#                        criterions=criterions,
#                        **config['Trainer'])

cotrainner.start_training(**config['StartTraining'])
