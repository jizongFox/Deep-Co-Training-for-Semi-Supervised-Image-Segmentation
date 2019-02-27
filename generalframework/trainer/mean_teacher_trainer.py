from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from generalframework.models import Segmentator
from .trainer import Trainer


class MeanTeacherTrainer(Trainer):
    def __init__(self,
                 student_segmentator: Segmentator,
                 teacher_segmentator: Segmentator,
                 labeled_dataloader: DataLoader,
                 unlabeled_dtaloader: DataLoader,
                 criterions: Dict[str,nn.Module],
                 max_epoch: int = 100,
                 save_dir: str = 'tmp',
                 device: str = 'cpu',
                 axises: List[int] = [1, 2, 3],
                 checkpoint: str = None,
                 metricname: str = 'metrics.csv',
                 whole_config=None
                 ) -> None:

        self.max_epoch = max_epoch
        self.teacher = teacher_segmentator
        self.student = student_segmentator
        self.labled_dataset
        self.criterions = criterions
        self.save_dir = Path(save_dir)
        # assert not (self.save_dir.exists() and checkpoint is None), f'>> save_dir: {self.save_dir} exits.'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if whole_config:
            with open(Path(self.save_dir, 'mt_config.yml'), 'w') as outfile:
                yaml.dump(whole_config, outfile, default_flow_style=True)
        self.device = torch.device(device)
        self.C = student_segmentator.arch_params['num_classes']
        self.axises = axises
        self.best_score = -1
        self.start_epoch = 0
        self.metricname = metricname

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        self.to(self.device)
