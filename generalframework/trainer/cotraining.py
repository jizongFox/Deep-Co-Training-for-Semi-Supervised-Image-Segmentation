from typing import Dict, List, Union
from pathlib import Path
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from generalframework import ModelMode
from ..models import Segmentator
from .trainer import Base, Trainer
from ..utils.utils import *
from ..loss import CrossEntropyLoss2d, KL_Divergence_2D
from copy import deepcopy as dcopy
from random import random
from functools import reduce

from tensorboardX import SummaryWriter


class CoTrainer(Trainer):

    def __init__(self, segmentators: List[Segmentator], labeled_dataloaders: List[DataLoader],
                 unlabeled_dataloader: DataLoader, val_dataloader: DataLoader, criterions: Dict[str, nn.Module],
                 max_epoch: int = 100, save_dir: str = 'tmp', device: str = 'cpu',
                 axises: List[int] = [0, 1, 2], checkpoint: str = None, metricname: str = 'metrics.csv',
                 lambda_cot_max: int = 10, lambda_adv_max: float = 0.5, ramp_up_mult: float = -5., epoch_max_ramp: int = 80) -> None:

        self.max_epoch = max_epoch
        self.segmentators = segmentators
        self.labeled_dataloaders = labeled_dataloaders
        self.unlabeled_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader
        self.lambda_cot_max = lambda_cot_max
        self.lambda_adv_max = lambda_adv_max
        self.epoch_max_ramp = epoch_max_ramp
        self.ramp_up_mult = ramp_up_mult

        ## N segmentators should be consist with N+1 dataloders
        # (N for labeled data and N+2 th for unlabeled dataset)
        assert self.segmentators.__len__() == self.labeled_dataloaders.__len__()
        assert self.segmentators.__len__() >= 1
        ## the sgementators and dataloaders must be different instance
        assert set(map_(id, self.segmentators)).__len__() == self.segmentators.__len__()
        assert set(map_(id, self.labeled_dataloaders)).__len__() == self.segmentators.__len__()

        ## labeled_dataloaders should have the same number of images
        # assert set(map_(lambda x: len(x.dataset), self.labeled_dataloaders)).__len__() == 1
        # assert set(map_(lambda x: len(x), self.labeled_dataloaders)).__len__() == 1

        self.criterions = criterions
        assert set(self.criterions.keys()) == set(['jsd', 'sup', 'adv'])

        self.save_dir = Path(save_dir)
        # assert not (self.save_dir.exists() and checkpoint is None), f'>> save_dir: {self.save_dir} exits.'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(save_dir)
        self.device = torch.device(device)
        self.C = self.segmentators[0].arch_params['num_classes']
        self.axises = axises
        self.best_score = -1
        self.start_epoch = 0
        self.metricname = metricname

        if checkpoint is not None:
            # todo
            self._load_checkpoint(checkpoint)

        self.to(self.device)

    def to(self, device: torch.device):
        [segmentator.to(device) for segmentator in self.segmentators]
        [criterion.to(device) for _, criterion in self.criterions.items()]

    def start_training(self, train_jsd=False, train_adv=False, save_train=False, save_val=False):
        ## prepare for something:

        for epoch in range(self.start_epoch + 1, self.max_epoch):
            self.schedulerStep()

            lambda_cot, lambda_adv = self.adjust_multipliers(self.lambda_cot_max, self.lambda_adv_max, self.ramp_up_mult,
                                                             epoch, epoch_max_ramp=self.epoch_max_ramp)

            train_lab_dice, train_unlab_dice = self._train_loop(labeled_dataloaders=self.labeled_dataloaders,
                                                                unlabeled_dataloader=self.unlabeled_dataloader,
                                                                epoch=epoch,
                                                                mode=ModelMode.TRAIN,
                                                                save=save_train,
                                                                train_jsd=train_jsd,
                                                                train_adv=train_adv
                                                                )

            with torch.no_grad():
                val_dice, val_batch_dice = self._eval_loop(val_dataloader=self.val_dataloader,
                                                           epoch=epoch,
                                                           mode=ModelMode.EVAL,
                                                           save=save_val)

    def _train_loop(self, labeled_dataloaders: List[DataLoader], unlabeled_dataloader: DataLoader, epoch: int,
                    mode: ModelMode, save: bool, augment_labeled_data=True, augment_unlabeled_data=False,
                    train_jsd=False, train_adv=False):
        [segmentator.set_mode(mode) for segmentator in self.segmentators]
        for l_dataloader in labeled_dataloaders:
            l_dataloader.dataset.set_mode(ModelMode.TRAIN)
        unlabeled_dataloader.dataset.training = ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        assert self.segmentators[0].training == True
        assert self.labeled_dataloaders[
                   0].dataset.training == ModelMode.TRAIN if augment_labeled_data else ModelMode.EVAL
        assert self.unlabeled_dataloader.dataset.training == ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        # Here the concept of epoch is defined as the epoch
        n_img = max(map_(lambda x: len(x.dataset), self.labeled_dataloaders))
        n_batch = max(map_(len, self.labeled_dataloaders))
        S = len(self.segmentators)
        # S labeled dataset + 1 unlabeled dataset
        n_unlab_img = n_batch * unlabeled_dataloader.batch_size

        coef_dice = torch.zeros(n_img, S, self.C)
        unlabel_coef_dice = torch.zeros(n_unlab_img, S, self.C)
        loss_log = torch.zeros(n_batch, S)

        lab_done = 0
        unlab_done = 0

        ## build fake_iterator
        fake_labeled_iterators = [iterator_(dcopy(x)) for x in labeled_dataloaders]
        fake_labeled_iterators_adv = [iterator_(dcopy(x)) for x in labeled_dataloaders]

        fake_unlabeled_iterator = iterator_(dcopy(unlabeled_dataloader))
        fake_unlabeled_iterator_adv = iterator_(dcopy(unlabeled_dataloader))

        n_batch_iter = tqdm_(range(n_batch))

        nice_dict = {}
        lab_dsc_dict = {}
        unlab_dsc_dict = {}
        report_iterator = iterator_(['label', 'unlab'])
        report_status = 'label'

        for batch_num in n_batch_iter:
            if batch_num % 30 == 0 and train_jsd:
                report_status = report_iterator.__next__()

            c_dices, sup_losses = [], []
            ## for labeled data update
            for enu_lab in range(len(fake_labeled_iterators)):
                [[img, gt], _, _] = fake_labeled_iterators[enu_lab].__next__()
                img, gt = img.to(self.device), gt.to(self.device)
                lab_B = img.shape[0]
                ## backward and update when the mode = ModelMode.TRAIN
                pred, sup_loss = self.segmentators[enu_lab].update(img, gt, criterion=self.criterions.get('sup'),
                                                                   mode=ModelMode.TRAIN if enu_lab==0 else ModelMode.EVAL)
                c_dice = dice_coef(*self.toOneHot(pred, gt))  # shape: B, axises
                c_dices.append(c_dice)
                sup_losses.append(sup_loss)

            batch_slice = slice(lab_done, lab_done + lab_B)
            ## record supervised data
            coef_dice[batch_slice] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)
            loss_log[batch_num] = torch.cat([x.unsqueeze(0) for x in sup_losses], dim=0)
            lab_done += lab_B
            # sup_loss = reduce(lambda x, y: x + y, sup_losses)
            if train_jsd:
                ## for unlabeled data update
                [[unlab_img, unlab_gt], _, _] = fake_unlabeled_iterator.__next__()
                unlab_B = unlab_img.shape[0]
                unlab_img, unlab_gt = unlab_img.to(self.device), unlab_gt.to(self.device)
                unlab_preds: List[Tensor] = map_(lambda x: x.predict(unlab_img, logit=False), self.segmentators)
                assert unlab_preds.__len__() == self.segmentators.__len__()

                c_dices = map_(lambda x: dice_coef(*self.toOneHot(x, unlab_gt)), unlab_preds)
                batch_slice = slice(unlab_done, unlab_done + unlab_B)
                ## record unlabeled data
                unlabel_coef_dice[batch_slice] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)

                ## function for JSD
                jsdloss_2D = self.criterions.get('jsd')(unlab_preds)
                assert jsdloss_2D.shape == unlab_img.squeeze(1).shape
                jsdloss = jsdloss_2D.mean()
                unlab_done += unlab_B

                ## backward and update
                # zero grad
                map_(lambda x: x.optimizer.zero_grad(), self.segmentators)
                loss = jsdloss
                loss.backward()
                map_(lambda x: x.optimizer.step(), self.segmentators)

            ## adversarial loss:

            if train_adv:
                assert self.segmentators.__len__() == 2, 'only implemented for 2 segmentators'
                ### TODO for more than 2 segmentators
                adv_losses = []

                ## draw first term from labeled1 or unlabeled
                img, img_adv = None, None
                if random() >= 1:
                    [[img, gt], _, _] = fake_labeled_iterators_adv[0].__next__()
                    img, gt = img.to(self.device), gt.to(self.device)
                    img_adv = fsgm_img_generator(self.segmentators[0].torchnet, eplision=0.05)(dcopy(img), gt,
                                                                                               criterion=CrossEntropyLoss2d())
                else:
                    [[img, _], _, _] = fake_unlabeled_iterator_adv.__next__()
                    img = img.to(self.device)
                    img_adv = VATGenerator(self.segmentators[0].torchnet, eplision=0.05)(dcopy(img))
                assert img.shape == img_adv.shape
                adv_pred = self.segmentators[1].predict(img_adv, logit=False)
                real_pred = self.segmentators[0].predict(img, logit=False)
                adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred))

                if random() > 0.5:
                    [[img, gt], _, _] = fake_labeled_iterators_adv[1].__next__()
                    img, gt = img.to(self.device), gt.to(self.device)
                    img_adv = fsgm_img_generator(self.segmentators[1].torchnet, eplision=0.05)(img, gt,
                                                                                               criterion=CrossEntropyLoss2d())
                else:
                    [[img, _], _, _] = fake_unlabeled_iterator_adv.__next__()
                    img = img.to(self.device)
                    img_adv = VATGenerator(self.segmentators[1].torchnet, eplision=0.05)(img)

                adv_pred = self.segmentators[0].predict(img_adv, logit=False)
                real_pred = self.segmentators[1].predict(img, logit=False)
                adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred))

                adv_loss = sum(adv_losses) / adv_losses.__len__()

                map_(lambda x: x.optimizer.zero_grad(), self.segmentators)
                adv_loss.backward()
                map_(lambda x: x.optimizer.step(), self.segmentators)

            lab_big_slice = slice(0, lab_done)
            unlab_big_slice = slice(0, unlab_done)

            lab_dsc_dict = {f"S{i}": {f"DSC{n}": coef_dice[lab_big_slice, i, n].mean().item() for n in self.axises} for
                            i in
                            range(len(self.segmentators))}
            unlab_dsc_dict = {f"S{i}": {f"DSC{n}": unlabel_coef_dice[unlab_big_slice, i, n].mean().item() \
                                        for n in self.axises} for i in range(len(self.segmentators))}

            lab_mean_dict = {f"S{i}": {"DSC": coef_dice[lab_big_slice, i, self.axises].mean().item()} for i in
                             range(len(self.segmentators))}

            unlab_mean_dict = {f"S{i}": {"DSC": unlabel_coef_dice[lab_big_slice, i, self.axises].mean().item()} for i in
                               range(len(self.segmentators))}

            # the general shape of the dict to save upload

            loss_dict = {f'L{i}': loss_log[0:batch_num, i].mean().item() for i in range(len(self.segmentators))}

            nice_dict = {k: {**v1, **v2} for k, v1 in lab_dsc_dict.items() for _, v2 in
                         lab_mean_dict.items()} if report_status == 'label' else \
                {k: {**v1, **v2} for k, v1 in unlab_dsc_dict.items() for _, v2 in unlab_mean_dict.items()}

            n_batch_iter.set_postfix({f'{k}_{k_}': f'{v[k_]:.2f}' for k, v in nice_dict.items() for k_ in v.keys()})
            n_batch_iter.set_description(
                report_status + ': ' + ','.join([f'{k}:{v:.3f}' for k, v in loss_dict.items()]))
            #
        self.upload_dicts('labeled dataset', lab_dsc_dict, epoch)
        self.upload_dicts('unlabeled dataset', unlab_dsc_dict, epoch)

        print(
            f"{desc} " + ', '.join([f'{k}_{k_}: {v[k_]:.2f}' for k, v in nice_dict.items() for k_ in v.keys()])
        )
        return coef_dice, unlabel_coef_dice

    def _eval_loop(self, val_dataloader: DataLoader,
                   epoch: int,
                   mode: ModelMode = ModelMode.EVAL,
                   save: bool = False
                   ):
        [segmentator.set_mode(mode) for segmentator in self.segmentators]
        val_dataloader.dataset.set_mode(ModelMode.EVAL)
        assert self.segmentators[0].training == False
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        S = self.segmentators.__len__()
        n_img = len(val_dataloader.dataset)
        n_batch = len(val_dataloader)
        coef_dice = torch.zeros(n_img, S, self.C)
        batch_dice = torch.zeros(n_batch, S, self.C)
        loss_log = torch.zeros(n_batch, S)
        val_dataloader = tqdm_(val_dataloader)
        done = 0

        dsc_dict = {}

        for batch_num, [(img, gt), _, _] in enumerate(val_dataloader):
            img, gt = img.to(self.device), gt.to(self.device)
            B = img.shape[0]
            preds = map_(lambda x: x.predict(img, logit=True), self.segmentators)
            c_dices = map_(lambda x: dice_coef(*self.toOneHot(x, gt)), preds)  # shape: B, axises
            b_dices = map_(lambda x: dice_batch(*self.toOneHot(x, gt)), preds)
            batch_slice = slice(done, done + B)
            coef_dice[batch_slice] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)
            batch_dice[batch_num] = torch.cat([x.unsqueeze(0) for x in b_dices], dim=0)
            done += B

            ## todo save predictions

            big_slice = slice(0, done)

            dsc_dict = {f"S{i}": {f"DSC{n}": coef_dice[big_slice, i, n].mean().item() for n in self.axises} \
                        for i in range(len(self.segmentators))}

            b_dsc_dict = {f"S{i}": {f"bDSC{n}": batch_dice[big_slice, i, n].mean().item() for n in self.axises} \
                          for i in range(len(self.segmentators))}

            mean_dict = {f"S{i}": {"DSC": coef_dice[big_slice, i, self.axises].mean().item()} for i in
                         range(len(self.segmentators))}

            nice_dict = {k: {**v1, **v2} for k, v1 in dsc_dict.items() for _, v2 in mean_dict.items()}

            loss_dict = {f'L{i}': loss_log[0:batch_num, i].mean().item() for i in range(len(self.segmentators))}
            val_dataloader.set_description('val: ' + ','.join([f'{k}:{v:.3f}' for k, v in loss_dict.items()]))

            val_dataloader.set_postfix({f'{k}_{k_}': f'{v[k_]:.2f}' for k, v in nice_dict.items() for k_ in v.keys()})

        self.upload_dicts('val_dataset', dsc_dict, epoch)
        print(
            f"{desc} " + ', '.join([f'{k}_{k_}: {v[k_]:.2f}' for k, v in nice_dict.items() for k_ in v.keys()])
        )
        return coef_dice, batch_dice

    def upload_dicts(self, name, dicts, epoch):
        for k, v in dicts.items():
            name_ = name + '/' + k
            self.upload_dict(name_, v, epoch)

    def upload_dict(self, name, dict, epoch):
        self.writer.add_scalars(name, dict, epoch)

    def schedulerStep(self, epoch,  n_labeled, n_samples):
        for segmentator in self.segmentators:
            segmentator.schedulerStep()

    def adjust_multipliers(self, lambda_cot_max, lambda_diff_max, ramp_up_mult, epoch, epoch_max_ramp):
        n_labeled = self.labeled_dataloaders[0].__len__()
        n_samples = n_labeled + self.unlabeled_dataloade.__len__()
        # this is the ramp_up function for lambda_cot and lambda_diff weights on the unsupervised terms.
        lambda_cot = weight_schedule(epoch, self.epoch_max_ramp, self.lambda_cot_max, self.ramp_up_mult,
                                     n_labeled, n_samples)
        lambda_adv = weight_schedule(epoch, self.epoch_max_ramp, self.lambda_adv_max, self.ramp_up_mult,
                                     n_labeled, n_samples)
        # turn it into a usable pytorch object
        lambda_cot = torch.FloatTensor([lambda_cot]).to(self.device)
        lambda_adv = torch.FloatTensor([lambda_adv]).to(self.device)
        return lambda_cot, lambda_adv
