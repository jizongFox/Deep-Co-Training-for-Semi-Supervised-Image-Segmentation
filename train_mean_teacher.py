from pprint import pprint
import warnings
import yaml
from easydict import EasyDict
from torch import nn

from generalframework.dataset.ACDC_helper import get_ACDC_split_dataloders
from generalframework.dataset.GM_helper import get_GMC_split_dataloders
from generalframework.models import Segmentator
from generalframework.trainer.mean_teacher_trainer import MeanTeacherTrainer
from generalframework.utils import yaml_parser, dict_merge

warnings.filterwarnings('ignore')

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config/ACDC_meanteacher_config.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
pprint(config)

config = EasyDict(config)
student = Segmentator(arch_dict=config.Arch, optim_dict=config.Optim, scheduler_dict=config.Scheduler)
# checkpoint = torch.load(
#     'archives/GM/gm/enet_adv_weigth_check0/FS/best_0.pth',
#     map_location='cpu')
# student.load_state_dict(checkpoint['segmentator'])
teacher = Segmentator(arch_dict=config.Arch, optim_dict=config.Optim, scheduler_dict=config.Scheduler)
# teacher.load_state_dict(student.state_dict)
# dataset

if config['Dataset']['root_dir'].find('ACDC') >= 0:
    labeled_dataloader, unlabeled_dataloader, val_dataloader = get_ACDC_split_dataloders(config)
elif config['Dataset']['root_dir'].find('GM') >= 0:

    labeled_dataloader, unlabeled_dataloader, val_dataloader = get_GMC_split_dataloders(config)
else:
    raise NotImplementedError

meanTeacherTrainer = MeanTeacherTrainer(
    student_segmentator=student,
    teacher_segmentator=teacher,
    labeled_dataloader=labeled_dataloader[0],
    unlabeled_dataloader=unlabeled_dataloader,
    val_dataloader=val_dataloader,
    criterions={'sup': nn.CrossEntropyLoss(),
                'con': nn.MSELoss()},
    cot_scheduler_dict=config['Cot_Scheduler'],
    **config['Trainer'],
    whole_config=config
)
meanTeacherTrainer.start_training()
