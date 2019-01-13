from typing import Dict, List, Union
from pathlib import Path
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from generalframework import ModelMode
from ..models import Segmentator
from .trainer import Base, Trainer
from ..utils.utils import *


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
            # with torch.no_grad():
            #     val_dice, val_batch_dice, val_loss = self._main_loop(val_loader, epoch, mode=ModelMode.EVAL,
            #                                                          save=self.save_val)
            # self.segmentator.schedulerStep()
            #
            # ## the saved metrics are with only axis==3, as the foreground dice.
            #
            # current_metric = val_dice[:, self.axises].mean()
            # # todo
            # self.checkpoint(current_metric, epoch)

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
        S = len(self.segmentators)
        # Here the concept of epoch is defined as the epoch
        n_img = len(self.labeled_dataloaders[0].dataset)
        n_batch = len(self.labeled_dataloaders[0])
        # S labeled dataset + 1 unlabeled dataset
        coef_dice = torch.zeros(n_img, S, self.C, 2)
        batch_dice = torch.zeros(n_batch, S, self.C, 2)
        loss_log = torch.zeros(n_batch, S + 1)
        done = 0

        ## build fake_iterator
        fake_labeled_iterators = [iterator_(x) for x in labeled_dataloaders]
        fake_unlabeled_iterator = iterator_(unlabeled_dataloader)
        n_batch_iter = tqdm_(range(n_batch))

        done = 0
        for i in n_batch_iter:

            c_dices, suploss = [], []

            ## for labeled data update
            for enu_lab in range(len(fake_labeled_iterators)):
                [[img, gt], _, path] = fake_labeled_iterators[enu_lab].__next__()
                img, gt = img.to(self.device), gt.to(self.device)
                B = img.shape[0]
                pred, sup_loss = self.segmentators[enu_lab].update(img, gt, criterion=self.criterions.get('sup'),
                                                                   mode=ModelMode.TRAIN)
                ohpredmask, ohmask = self.toOneHot(pred, gt)
                c_dice = dice_coef(ohpredmask, ohmask)

                c_dices.append(c_dice)

            ## record supervised data
            batch_slice = slice(done, done + B)
            coef_dice[batch_slice][..., 0] = torch.cat([x.unsqueeze(1) for x in c_dices], dim=1)

            ## for unlabeled data update
            [[unlab_img, _], _, path] = fake_unlabeled_iterator.__next__()
            unlab_img = unlab_img.to(self.device)
            unlab_preds: List[Tensor] = map_(lambda x: x.predict(unlab_img, logit=False), self.segmentators)
            assert unlab_preds.__len__() == self.segmentators.__len__()

            ## function for JSD
            # todo
            pred, loss = self.criterions.get('jsd')(unlab_preds)

            ## record unlabeled data



            done += B

            big_slice = slice(0, done)

            dsc_dict = {f"S{i}:DSC{n}": coef_dice[big_slice, i, n, 0].mean() for i in
                        range(len(self.segmentators)) for n in self.axises}
            mean_dict = {f"S{i}:DSC": coef_dice[big_slice, i, self.axises, 0].mean() for i in
                         range(len(self.segmentators))}

            nice_dict = {**dsc_dict, **mean_dict}

            n_batch_iter.set_description(''.join(
                ['%s:%.3f ' % (k, v.item()) for k, v in {k: v for k, v in nice_dict.items() if v != 0}.items()]))
            #
        print('finished one epoch')
        print(f"{desc} " + ', '.join(f"{k}={v:.3f}" for (k, v) in nice_dict.items()))

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
