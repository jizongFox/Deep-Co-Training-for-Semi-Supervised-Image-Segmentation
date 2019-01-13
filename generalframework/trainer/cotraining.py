from typing import Dict, List, Union
from pathlib import Path
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from generalframework import ModelMode
from ..models import Segmentator
from .trainer import Base, Trainer
from ..utils.utils import *
from copy import deepcopy as dcopy
from random import random
from functools import reduce


class CoTrainer(Trainer):

    def __init__(self, segmentators: List[Segmentator], labeled_dataloaders: List[DataLoader],
                 unlabeled_dataloader: DataLoader, val_dataloader: DataLoader, criterions: Dict[str, nn.Module],
                 max_epoch: int = 100, save_dir: str = 'tmp', save_train=False, save_val=False, device: str = 'cpu',
                 axises: List[int] = [0, 1, 2], checkpoint: str = None, metricname: str = 'metrics.csv') -> None:

        self.max_epoch = max_epoch
        self.segmentators = segmentators
        self.labeled_dataloaders = labeled_dataloaders
        self.unlabeled_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader

        ## N segmentators should be consist with N+1 dataloders
        # (N for labeled data and N+2 th for unlabeled dataset)
        assert self.segmentators.__len__() == self.labeled_dataloaders.__len__()
        assert self.segmentators.__len__() >= 1
        ## the sgementators and dataloaders must be different instance
        assert set(map_(id, self.segmentators)).__len__() == self.segmentators.__len__()
        assert set(map_(id, self.labeled_dataloaders)).__len__() == self.segmentators.__len__()

        ## labeled_dataloaders should have the same number of images
        assert set(map_(lambda x: len(x.dataset), self.labeled_dataloaders)).__len__() == 1
        assert set(map_(lambda x: len(x), self.labeled_dataloaders)).__len__() == 1

        self.criterions = criterions
        assert set(self.criterions.keys()) == set(['jsd', 'sup', 'adv'])
        print('\nThe losses used are:')
        for k, v in self.criterions.items():
            print('%s:' % k, v)

        self.save_dir = Path(save_dir)
        # assert not (self.save_dir.exists() and checkpoint is None), f'>> save_dir: {self.save_dir} exits.'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_train = save_train
        self.save_val = save_val
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

    def start_training(self):
        ## prepare for something:

        for epoch in range(self.start_epoch + 1, self.max_epoch):
            self._train_loop(labeled_dataloaders=self.labeled_dataloaders,
                             unlabeled_dataloader=self.unlabeled_dataloader,
                             epoch=epoch,
                             mode=ModelMode.TRAIN,
                             save=self.save_train)

    @staticmethod
    def record_batch_oneset():
        pass

    def _train_loop(self, labeled_dataloaders: List[DataLoader], unlabeled_dataloader: DataLoader, epoch: int,
                    mode: ModelMode, save: bool, augment_labeled_data=True, augment_unlabeled_data=False):
        [segmentator.set_mode(mode) for segmentator in self.segmentators]
        for l_dataloader in labeled_dataloaders:
            l_dataloader.dataset.set_mode(ModelMode.TRAIN if augment_labeled_data else ModelMode.EVAL)
        unlabeled_dataloader.dataset.training = ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        assert self.segmentators[0].training == True
        assert self.labeled_dataloaders[
                   0].dataset.training == ModelMode.TRAIN if augment_labeled_data else ModelMode.EVAL
        assert self.unlabeled_dataloader.dataset.training == ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        # Here the concept of epoch is defined as the epoch
        n_img = len(self.labeled_dataloaders[0].dataset)
        n_batch = len(self.labeled_dataloaders[0])
        S = len(self.segmentators)
        # S labeled dataset + 1 unlabeled dataset
        n_unlab_img = n_batch * unlabeled_dataloader.batch_size

        coef_dice = torch.zeros(n_img, S, self.C)
        unlabel_coef_dice = torch.zeros(n_unlab_img, S, self.C)

        loss_log = torch.zeros(n_batch, S + 1)
        lab_done = 0
        unlab_done = 0

        ## build fake_iterator
        fake_labeled_iterators = [iterator_(dcopy(x)) for x in labeled_dataloaders]
        fake_unlabeled_iterator = iterator_(dcopy(unlabeled_dataloader))
        n_batch_iter = tqdm_(range(n_batch))

        nice_dict = {}
        for i in n_batch_iter:

            c_dices, sup_losses = [], []

            ## for labeled data update
            for enu_lab in range(len(fake_labeled_iterators)):
                [[img, gt], _, _] = fake_labeled_iterators[enu_lab].__next__()
                img, gt = img.to(self.device), gt.to(self.device)
                lab_B = img.shape[0]
                pred, sup_loss = self.segmentators[enu_lab].update(img, gt, criterion=self.criterions.get('sup'),
                                                                   mode=ModelMode.TRAIN, is_backward=False)
                c_dice = dice_coef(*self.toOneHot(pred, gt))  # shape: B, axises
                c_dices.append(c_dice)
                sup_losses.append(sup_loss)

            batch_slice = slice(lab_done, lab_done + lab_B)
            ## record supervised data
            coef_dice[batch_slice] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)
            lab_done += lab_B
            sup_loss = reduce(lambda x, y: x + y, sup_losses)

            ## for unlabeled data update
            [[unlab_img, unlab_gt], _, _] = fake_unlabeled_iterator.__next__()
            unlab_B = unlab_img.shape[0]
            unlab_img, unlab_gt = unlab_img.to(self.device), unlab_gt.to(self.device)
            unlab_preds: List[Tensor] = map_(lambda x: x.predict(unlab_img, logit=False), self.segmentators)
            assert unlab_preds.__len__() == self.segmentators.__len__()

            c_dices = map_(lambda x: dice_coef(*self.toOneHot(x, unlab_gt)), unlab_preds)
            batch_slice = slice(unlab_done, unlab_done + unlab_B)
            unlabel_coef_dice[batch_slice] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)

            ## function for JSD
            jsdloss_2D = self.criterions.get('jsd')(unlab_preds)
            assert jsdloss_2D.shape == unlab_img.squeeze(1).shape
            jsdloss = jsdloss_2D.mean()
            unlab_done += unlab_B

            ## backward and update
            # zero grad
            map_(lambda x: x.optimizer.zero_grad(), self.segmentators)

            loss = sup_loss + jsdloss
            loss.backward()
            map_(lambda x: x.optimizer.step(), self.segmentators)

            ## record unlabeled data

            lab_big_slice = slice(0, lab_done)
            unlab_big_slice = slice(0, unlab_done)

            lab_dsc_dict = {f"S{i}": {f"DSC{n}": coef_dice[lab_big_slice, i, n].mean().item() for n in self.axises} for
                            i in
                            range(len(self.segmentators))}
            unlab_dsc_dict = {f"S{i}": {f"DSC{n}": unlabel_coef_dice[unlab_big_slice, i, n].mean().item() \
                                        for n in self.axises} for i in range(len(self.segmentators))}

            lab_mean_dict = {f"S{i}": {"DSC": coef_dice[lab_big_slice, i, self.axises].mean().item()} for i in
                             range(len(self.segmentators))}

            unlab_mean_dict = {f"S{i}": {"DSC": coef_dice[lab_big_slice, i, self.axises].mean().item()} for i in
                               range(len(self.segmentators))}

            nice_dict = lab_dsc_dict

            n_batch_iter.set_description(
                ','.join([f'{k}{k_}:{v[k_]:.3f}' for k, v in nice_dict.items() for k_ in v.keys()]))
            #
        # print('finished one epoch')
        print(
            f"{desc} " + ', '.join(','.join([f'{k}{k_}:{v[k_]:.3f}' for k, v in nice_dict.items() for k_ in v.keys()])))
        return coef_dice, unlabel_coef_dice

        #
        # for i, (imgs, metainfo, filenames) in enumerate(dataloader):
        #     B = filenames.__len__()
        #     imgs = [img.to(self.device) for img in imgs]
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", category=UserWarning)
        #         [preds, loss] = self.segmentator.update(imgs[0], imgs[1], self.criterion, mode=mode)
        #     ohpredmask, ohmask = self.toOneHot(preds, imgs[1])
        #     c_dice = dice_coef(ohpredmask, ohmask)
        #
        #     if mode == ModelMode.EVAL:
        #         b_dice = dice_batch(ohpredmask, ohmask)
        #
        #     batch = slice(done, done + B)
        #     coef_dice[batch] = c_dice
        #
        #     if mode == ModelMode.EVAL:
        #         batch_dice[batch] = b_dice
        #
        #     loss_log[batch] = loss
        #     done += B
        #
        #     if save:
        #         save_images(segs=preds.max(1)[1], names=filenames, root=self.save_dir, mode=mode.value.lower(),
        #                     iter=epoch)
        #
        #     ## for visualization
        #     big_slice = slice(0, done)  # Value for current and previous batches
        #
        #     dsc_dict = {f"DSC{n}": coef_dice[big_slice, n].mean() for n in self.axises}
        #
        #     bdsc_dict = {f"bDSC{n}": batch_dice[big_slice, n].mean() for n in
        #                  self.axises}
        #
        #     mean_dict = {"DSC": coef_dice[big_slice, self.axises].mean()}
        #
        #     stat_dict = {**dsc_dict, **bdsc_dict, **mean_dict,
        #                  "loss": loss_log[:i].mean()}
        #     # to delete null dicts
        #     nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}
        #
        #     dataloader.set_postfix(nice_dict)  ## using average value of the dict
        #
        # print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))
        #
        # return coef_dice, batch_dice, loss_log
