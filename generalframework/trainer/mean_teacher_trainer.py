import warnings
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import DataLoader
from numpy import array, uint32
import numpy as np
from generalframework.models import Segmentator
from .trainer import Trainer
from .. import ModelMode
from ..metrics import DiceMeter, AggragatedMeter, AverageValueMeter
from ..utils import tqdm_,iterator_
from ..dataset.augment import TensorAugment,temporary_seed

class MeanTeacherTrainer(Trainer):
    def __init__(self,
                 student_segmentator: Segmentator,
                 teacher_segmentator: Segmentator,
                 labeled_dataloader: DataLoader,
                 unlabeled_dataloader: DataLoader,
                 val_dataloader:DataLoader,
                 criterions: Dict[str, nn.Module],
                 max_epoch: int = 100,
                 save_dir: str = 'tmp',
                 device: str = 'cuda:0',
                 axises: List[int] = [1, 2, 3],
                 checkpoint: str = None,
                 metricname: str = 'metrics.csv',
                 whole_config=None
                 ) -> None:

        self.max_epoch = max_epoch
        self.teacher = teacher_segmentator
        self.student = student_segmentator
        self.labeled_dataloader = labeled_dataloader
        self.unlabel_dataloader = unlabeled_dataloader
        self.val_dataloader= val_dataloader
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

    def to(self, device):
        self.teacher.to(device)
        self.student.to(device)

    def _load_checkpoint(self, checkpoint):
        '''
        :param checkpoint: checkpoint list for student and teacher for neither best or last
        :return: well loaded student and teacher networks.
        '''
        # the load_checkpoint must be rewritten
        pass

    def checkpoint(self, metric, epoch, filename='best.pth'):
        s_state = self.student.state_dict
        t_state = self.teacher.state_dict

    def start_training(self, save_train=False, save_val=False):
        self.METERS = edict()
        self.METERS.train_loss = AggragatedMeter(AverageValueMeter(), save_dir=self.save_dir)
        self.METERS.val_loss = AggragatedMeter(AverageValueMeter(), save_dir=self.save_dir)
        self.METERS.train_2D_dice = AggragatedMeter(DiceMeter(method='2d', C=self.C, report_axises=[1, 2, 3]),
                                                    save_dir=self.save_dir)
        self.METERS.val_2D_dice = AggragatedMeter(DiceMeter(method='2d', C=self.C, report_axises=[1, 2, 3]),
                                                  save_dir=self.save_dir)
        self.METERS.val_2D_dice = AggragatedMeter(DiceMeter(method='3d', C=self.C, report_axises=[1, 2, 3]),
                                                  save_dir=self.save_dir)

        self._train_loop(self.labeled_dataloader, self.unlabel_dataloader, epoch=1, mode=ModelMode.TRAIN)

        with torch.no_grad():
            self._eval_loop(self.val_dataloader, epoch=1)
        for k, v in self.METERS.items():
            v.Step()

    def _train_loop(self, labeled_dataloader, unlabeled_dataloader, epoch, mode=ModelMode.TRAIN):
        self.student.set_mode(mode)
        ## detach the teacher model
        self.teacher.set_mode(mode)
        self.labeled_dataloader.dataset.training=True
        self.unlabel_dataloader.dataset.training=True
        for param in self.teacher.torchnet.parameters():
            param.detach_()

        labeled_dataloader.dataset.set_mode(mode)
        unlabeled_dataloader.dataset.set_mode(mode)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        fake_unlabel_iter = iterator_(unlabeled_dataloader)
        dataloader_ = tqdm_(labeled_dataloader)
        for i, ((img, gt), ((o_img, o_gt), str_seed), filenames) in enumerate(dataloader_):
            img, gt, o_img, o_gt = img.to(self.device), gt.to(self.device), o_img.to(self.device), o_gt.to(self.device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                s_preds = self.student.predict(img,logit=True)
                sup_loss = self.criterions.get('sup')(s_preds,gt.squeeze(1))
                s_preds = F.softmax(s_preds,1)
            self.METERS.train_loss.Add(sup_loss.item())
            self.METERS.train_2D_dice.Add(s_preds,gt)
            with torch.no_grad():
                t_preds= self.teacher.predict(o_img,logit=False)
            img_lists,seed = zip(t_preds.detach()),map(eval,str_seed)
            t_preds_aug = []
            for i, (t_pred, seed) in enumerate(zip(img_lists,seed)):
                with temporary_seed(*seed):
                    t_preds_aug.append(TensorAugment(t_pred)[0])
            t_preds_aug=torch.Tensor(t_preds_aug).float().to(self.device)
            assert s_preds.shape == t_preds_aug.shape
            con_loss1 = self.criterions.get('con')(s_preds,t_preds_aug.detach())

            ((img, _), ((o_img, _), str_seed), filenames) = fake_unlabel_iter.__next__()
            img, o_img = img.to(self.device),  o_img.to(self.device)
            s_preds = self.student.predict(img, logit=False)
            with torch.no_grad():
                t_preds = self.teacher.predict(o_img, logit=False)
            img_lists,seed = zip(t_preds.detach()),map(eval,str_seed)
            t_preds_aug = []
            for i, (t_pred, seed) in enumerate(zip(img_lists,seed)):
                with temporary_seed(*seed):
                    t_preds_aug.append(TensorAugment(t_pred)[0])
            t_preds_aug = torch.Tensor(t_preds_aug).float().to(self.device)
            con_loss2 = self.criterions.get('con')(s_preds,t_preds_aug)
            total_loss = sup_loss+con_loss1+con_loss2
            self.student.optimizer.zero_grad()
            total_loss.backward()
            self.student.optimizer.step()
            self.update_ema()
            dataloader_.set_postfix({'loss':self.METERS.train_loss.Summary(),'dice':self.METERS.train_2D_dice.Summary()})
        print(self.METERS.train_loss.Summary())
        print(self.METERS.train_2D_dice.Summary())
        print(desc, {'train_loss':self.METERS.train_loss.Summary(),'dice':self.METERS.train_2D_dice.Summary()})

    def _eval_loop(self,val_dataloader,epoch, mode=ModelMode.EVAL):
        val_dataloader.dataset.training=False
        self.teacher.set_mode(mode)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"

        for i, ((img, gt), (_, _), filenames) in enumerate(val_dataloader):
            img,gt = img.to(self.device),gt.to(self.device)
            t_preds = self.teacher.predict(img,logit=True)
            val_loss = self.criterions.get('sup')(t_preds,gt.squeeze(1))
            self.METERS.val_loss.Add(val_loss.item())
            self.METERS.val_2D_dice.Add(t_preds,gt)
        print(desc, {'val_loss':self.METERS.val_loss.Summary(),'dice':self.METERS.val_2D_dice.Summary()})



    def update_ema(self,alpha=0.99):
        for ema_param, param in zip(self.teacher.torchnet.parameters(), self.student.torchnet.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)