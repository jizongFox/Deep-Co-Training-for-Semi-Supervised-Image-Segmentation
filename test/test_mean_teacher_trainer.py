seg_config = {
    'Arch':
        {'name': 'enet',
         'num_classes': 4},

    'Optim': {'name': 'Adam',
              'lr': 0.0005},

    'Scheduler': {'name': 'MultiStepLR',
                  'milestones': [10, 20, 30, 40, 50, 60, 70, 80, 90],
                  'gamma': 0.7},

    'Dataset': {'root_dir': '../dataset/ACDC-all', 'subfolders': ['img', 'gt'],

                'transform': 'segment_transform((256,256))',
                'augment': 'PILaugment',
                'pin_memory': False,
                'metainfo': ['getImage_GT', {'foldernames': ['img', 'gt']}]  ## important for mean teacher
                },

    'Dataloader': {
        'pin_memory': False,
        'batch_size': 3,
        'num_workers': 3,
        'shuffle': True,
        'drop_last': True,
    },

    'Lab_Partitions': {
        'label': [[1, 51]],
        'unlabel': [51, 101]
    }
}

import torch
from easydict import EasyDict
from torch import nn

from generalframework.dataset import get_ACDC_dataloaders, extract_patients
from generalframework.models import Segmentator


def get_dataloders(config):
    dataloders = get_ACDC_dataloaders(config['Dataset'], config['Dataloader'])
    labeled_dataloaders = []
    for i in config['Lab_Partitions']['label']:
        labeled_dataloaders.append(extract_patients(dataloders['train'], [str(x) for x in range(*i)]))

    unlab_dataloader = get_ACDC_dataloaders(config['Dataset'], config['Dataloader'], quite=True)['train']
    unlab_dataloader = extract_patients(unlab_dataloader, [str(x) for x in range(*config['Lab_Partitions']['unlabel'])])
    val_dataloader = dataloders['val']
    return labeled_dataloaders[0], unlab_dataloader, val_dataloader


config = EasyDict(seg_config)
student = Segmentator(arch_dict=config.Arch, optim_dict=config.Optim, scheduler_dict=config.Scheduler)
print(student)
checkpoint = torch.load(
    '../archives/cardiac/IMPORTANT_GRID_SEARCH/IMPORTANTgridsearch_cotraining_results_0.5/enet_search_group1/FS/best_1.pth',
    map_location='cpu')
student.load_state_dict(checkpoint['segmentator'])
student.train()
teacher = Segmentator(arch_dict=config.Arch, optim_dict=config.Optim, scheduler_dict=config.Scheduler)
teacher.load_state_dict(student.state_dict)
## dataset
labeled_dataloader, unlabeled_dataloader, val_dataloader = get_dataloders(config)
from generalframework.trainer.mean_teacher_trainer import MeanTeacherTrainer

meanTeacherTrainer = MeanTeacherTrainer(student_segmentator=student,
                                        teacher_segmentator=teacher,
                                        labeled_dataloader=labeled_dataloader,
                                        unlabeled_dataloader=unlabeled_dataloader,
                                        criterions={'sup': nn.CrossEntropyLoss(),
                                                    'con':nn.MSELoss()})
meanTeacherTrainer.start_training()