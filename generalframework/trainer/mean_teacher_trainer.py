import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import DataLoader

from generalframework.models import Segmentator
from .trainer import Trainer
from .. import ModelMode
from ..dataset.augment import temporary_seed, TensorAugment_4_dim
from ..metrics2 import DiceMeter, AggragatedMeter, AverageValueMeter, ListAggregatedMeter
from ..scheduler import *
from ..utils import tqdm_, iterator_, flatten_dict


class MeanTeacherTrainer(Trainer):
    def __init__(self,
                 student_segmentator: Segmentator,
                 teacher_segmentator: Segmentator,
                 labeled_dataloader: DataLoader,
                 unlabeled_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 criterions: Dict[str, nn.Module],
                 max_epoch: int = 100,
                 save_dir: str = 'tmp',
                 device: str = 'cuda:0',
                 axises: List[int] = [1, 2, 3],
                 cot_scheduler_dict: dict = {},
                 checkpoint: str = None,
                 metricname: str = 'metrics.csv',
                 whole_config=None
                 ) -> None:

        self.max_epoch = max_epoch
        self.teacher = teacher_segmentator
        self.student = student_segmentator
        self.labeled_dataloader = labeled_dataloader
        self.unlabel_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader
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
            self.checkpoint = checkpoint
            self._load_checkpoint(checkpoint)

        self.to(self.device)

        self.cot_scheduler: RampScheduler = eval(cot_scheduler_dict['name'])(
            **{k: v for k, v in cot_scheduler_dict.items() if k != 'name'})

    def to(self, device):
        self.teacher.to(device)
        self.student.to(device)

    def _load_checkpoint(self, checkpoint):
        '''
        :param checkpoint: checkpoint list for student and teacher for neither best or last
        :return: well loaded student and teacher networks.
        '''
        # the load_checkpoint must be rewritten
        checkpoint = torch.load(checkpoint, map_location='cpu')
        self.teacher.load_state_dict(checkpoint['teacher'])
        self.student.load_state_dict(checkpoint['student'])
        self.best_score = checkpoint['metric']
        self.start_epoch = checkpoint['epoch']
        print(f'Checkpoint has been loaded with success..\n'
              f'Best record: {self.best_score} @ {self.start_epoch} epoch')

    def save_checkpoint(self, metric, epoch, meters):
        s_state = self.student.state_dict
        t_state = self.teacher.state_dict
        meter_state = meters.state_dict
        last_checkpoint = {'student': s_state,
                           'teacher': t_state,
                           'metric': metric,
                           'epoch': epoch,
                           'meters': meter_state}
        torch.save(last_checkpoint, Path(self.save_dir, 'last.pth'))

        if isinstance(self.best_score, torch.Tensor) and isinstance(metric, torch.Tensor):
            metric = metric.to(self.best_score.device)
        if self.best_score < metric:
            self.best_score = metric
            [os.remove(p) for p in Path(self.save_dir).glob('best_*.pth')]
            shutil.copy(Path(self.save_dir, 'last.pth'), Path(self.save_dir, f'best_{metric}_{epoch}.pth'))

    def start_training(self, checkpoint=None):
        METERS = edict()
        METERS.tra_student_loss = AggragatedMeter()
        METERS.tra_teacher_2d_dice = AggragatedMeter()
        METERS.tra_stduent_2d_dice = AggragatedMeter()
        METERS.val_teacher_2d_dice = AggragatedMeter()
        METERS.val_teacher_3d_dice = AggragatedMeter()
        wholeMeter = ListAggregatedMeter(names=list(METERS.keys()), listAggregatedMeter=list(METERS.values()))
        Path(self.save_dir, 'meters').mkdir(exist_ok=True)
        if isinstance(getattr(self, 'checkpoint'), (str, Path)):
            wholeMeter.load_state_dict(torch.load(self.checkpoint, map_location='cpu')['meters'])

        for epoch in range(self.start_epoch, self.max_epoch):
            tra_teacher_2d_dice, tra_stduent_2d_dice, \
            tra_student_loss = self._train_loop(self.labeled_dataloader,
                                                self.unlabel_dataloader,
                                                epoch=epoch,
                                                mode=ModelMode.TRAIN)

            with torch.no_grad():
                save_criterion, val_teacher_2d_dice, \
                val_teacher_3d_dice, val_teacher_loss = self._eval_loop(self.val_dataloader, epoch=epoch)
            for k, v in METERS.items():
                v.add(eval(k))
            for k, v in METERS.items():
                v.summary().to_csv(Path(self.save_dir, 'meters', f'{k}.csv'))
            wholeMeter.summary().to_csv(Path(self.save_dir, f'wholeMeter.csv'))
            self.schedulerStep()
            self.save_checkpoint(save_criterion, epoch, wholeMeter)

    def _train_loop(self, labeled_dataloader, unlabeled_dataloader, epoch, mode=ModelMode.TRAIN):
        self.student.set_mode(mode)
        # detach the teacher model
        self.teacher.set_mode(mode)
        self.labeled_dataloader.dataset.training = True
        self.unlabel_dataloader.dataset.training = True
        for param in self.teacher.torchnet.parameters():
            param.detach_()

        labeled_dataloader.dataset.set_mode(mode)
        unlabeled_dataloader.dataset.set_mode(mode)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        fake_unlabel_iter = iterator_(unlabeled_dataloader)
        dataloader_ = tqdm_(labeled_dataloader)

        student_cDice = DiceMeter(method='2d', report_axises=self.axises, C=self.C)
        teacher_cDice = DiceMeter(method='2d', report_axises=self.axises, C=self.C)
        student_sup_loss_meter = AverageValueMeter()
        total_loss_meter = AverageValueMeter()

        for i, ((img, gt), ((o_img, o_gt), str_seed), filenames) in enumerate(dataloader_):
            img, gt, o_img, o_gt = img.to(self.device), gt.to(self.device), o_img.to(self.device), o_gt.to(self.device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                s_preds = self.student.predict(img, logit=True)
                sup_loss = self.criterions.get('sup')(s_preds, gt.squeeze(1))
                s_preds = F.softmax(s_preds, 1)
            student_cDice.add(s_preds, gt)
            student_sup_loss_meter.add(sup_loss.item())
            with torch.no_grad():
                t_preds = self.teacher.predict(o_img, logit=False)
            teacher_cDice.add(t_preds, o_gt)
            pred_lists, seed = zip(t_preds.detach()), map(eval, str_seed)
            t_preds_aug = []
            for i, (t_pred, seed) in enumerate(zip(pred_lists, seed)):
                with temporary_seed(*seed):
                    t_preds_aug.append(TensorAugment_4_dim(t_pred)[0])
            t_preds_aug = torch.Tensor(t_preds_aug).float().to(self.device)

            assert s_preds.shape == t_preds_aug.shape

            con_loss1 = self.criterions.get('con')(s_preds, t_preds_aug.detach())
            ((img, _), ((o_img, _), str_seed), filenames) = fake_unlabel_iter.__next__()
            img, o_img = img.to(self.device), o_img.to(self.device)
            s_preds = self.student.predict(img, logit=False)
            with torch.no_grad():
                t_preds = self.teacher.predict(o_img, logit=False)
            img_lists, seed = zip(t_preds.detach()), map(eval, str_seed)
            t_preds_aug = []
            for i, (t_pred, seed) in enumerate(zip(img_lists, seed)):
                with temporary_seed(*seed):
                    t_preds_aug.append(TensorAugment_4_dim(t_pred)[0])
            t_preds_aug = torch.Tensor(t_preds_aug).float().to(self.device)

            #
            # import matplotlib.pyplot as plt
            #
            # plt.figure(1)
            # plt.clf()
            # plt.title('augmented image and original image')
            # plt.subplot(121)
            # plt.imshow(img[0].squeeze().cpu())
            # plt.subplot(122)
            # plt.imshow(o_img[0].squeeze().cpu())
            #
            # plt.subplot(121)
            # plt.imshow(t_preds_aug.max(1)[1][0].squeeze().cpu(),alpha=0.5)
            # plt.subplot(122)
            # plt.imshow(t_preds.max(1)[1][0].squeeze().cpu(),alpha=0.5)
            # plt.show()
            # plt.pause(0.5)

            con_loss2 = self.criterions.get('con')(s_preds, t_preds_aug)
            total_loss = sup_loss + self.cot_scheduler.value * (con_loss1 + con_loss2)
            total_loss_meter.add(total_loss.item())
            self.student.optimizer.zero_grad()
            total_loss.backward()
            self.student.optimizer.step()
            self.update_ema()

            report_dict = {'t': teacher_cDice.summary(),
                           's': student_cDice.summary()}
            report_dict = flatten_dict(report_dict, sep='')
            dataloader_.set_postfix(report_dict)
            dataloader_.set_description(f'tsloss:{student_sup_loss_meter.summary()["mean"]:.3f}')
        print(f'{desc} {", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])}')
        return teacher_cDice.detailed_summary(), student_cDice.detailed_summary(), student_sup_loss_meter.detailed_summary()

    def _eval_loop(self, val_dataloader, epoch, mode=ModelMode.EVAL):
        val_dataloader.dataset.set_mode(mode)
        assert val_dataloader.dataset.training == ModelMode.EVAL
        self.teacher.set_mode(mode)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        val_dataloader_ = tqdm_(val_dataloader)

        teacher_2D_dice = DiceMeter(method='2d', report_axises=self.axises, C=self.C)
        teacher_3D_dice = DiceMeter(method='3d', report_axises=self.axises, C=self.C)
        teacher_loss = AverageValueMeter()

        for i, ((img, gt), (_, _), filenames) in enumerate(val_dataloader_):
            img, gt = img.to(self.device), gt.to(self.device)
            t_preds = self.teacher.predict(img, logit=True)
            val_loss = self.criterions.get('sup')(t_preds, gt.squeeze(1))
            teacher_2D_dice.add(t_preds, gt)
            teacher_3D_dice.add(t_preds, gt)
            teacher_loss.add(val_loss.item())
            report_dict = {
                'c': teacher_2D_dice.summary(),
                'b': teacher_3D_dice.summary()
            }
            report_dict = flatten_dict(report_dict, sep='')
            val_dataloader_.set_description(f'vtloss:{teacher_loss.summary()["mean"]:.3f}')
            val_dataloader_.set_postfix(report_dict)

        print(f'{desc} {", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])}')
        return teacher_2D_dice.value()[0][
                   0], teacher_2D_dice.detailed_summary(), teacher_3D_dice.detailed_summary(), teacher_loss.detailed_summary()

    def update_ema(self, alpha=0.99):
        for ema_param, param in zip(self.teacher.torchnet.parameters(), self.student.torchnet.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def schedulerStep(self):
        self.student.scheduler.step()
        self.cot_scheduler.step()
