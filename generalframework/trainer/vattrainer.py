import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import yaml
from generalframework import ModelMode
from ..utils import *
from ..models import Segmentator
from .trainer import Trainer
from ..scheduler import RampScheduler
from ..loss import KL_Divergence_2D


class VatTrainer(Trainer):
    def __init__(self, segmentator: Segmentator, dataloaders: Dict[str, DataLoader], criterion: nn.Module,
                 max_epoch: int = 100,
                 save_dir: str = 'tmp', save_train=False, save_val=False,
                 device: str = 'cpu', axises: List[int] = [1, 2, 3],
                 checkpoint: str = None, metricname: str = 'metrics.csv', whole_config=None,
                 epoch_max_ramp: int = 80, lambda_adv_max: float = 0.5, ramp_up_mult=-5) -> None:
        super().__init__(segmentator=segmentator, dataloaders=dataloaders, criterion=criterion,
                         max_epoch=max_epoch, save_dir=save_dir,
                         device=device, axises=axises, checkpoint=checkpoint, metricname=metricname,
                         whole_config=whole_config)
        self.adv_scheduler = RampScheduler(max_epoch=epoch_max_ramp, max_value=lambda_adv_max, ramp_mult=ramp_up_mult)

    def start_training(self, train_adv=False, save_train=False, save_val=False):
        n_class: int = self.C
        train_b: int = len(self.dataloaders['lab'])  # Number of iteration per epoch: different if batch_size > 1
        train_n = train_b * self.dataloaders['lab'].batch_size if self.dataloaders['lab'].drop_last else len(
            self.dataloaders['lab'].dataset)  # when the droplast has been enabled.
        n_unlab_img = train_b * self.dataloaders['unlab'].batch_size if self.dataloaders['unlab'].drop_last else len(
            self.dataloaders['unlab'].dataset)
        val_b: int = len(self.dataloaders['val'])
        val_n: int = val_b * self.dataloaders['val'].batch_size if self.dataloaders['val'].drop_last == True \
            else len(self.dataloaders['val'].dataset)

        metrics = {"val_dice": torch.zeros((self.max_epoch, val_n, 1, n_class), device=self.device).type(torch.float32),
                   "val_batch_dice": torch.zeros((self.max_epoch, val_b, 1, n_class), device=self.device).type(
                       torch.float32),
                   "train_dice": torch.zeros((self.max_epoch, train_n, 1, n_class), device=self.device).type(
                       torch.float32),
                   "train_unlab_dice": torch.zeros((self.max_epoch, n_unlab_img, 1, n_class), device=self.device).type(
                       torch.float32),
                   "train_loss": torch.zeros((self.max_epoch, train_b, 1), device=self.device).type(torch.float32)}

        for epoch in range(self.start_epoch, self.max_epoch):
            train_dice, train_unlab_dice, train_loss = self._train_loop(labeled_dataloader=self.dataloaders['lab'],
                                                                        unlabeled_dataloader=self.dataloaders['unlab'],
                                                                        epoch=epoch, mode=ModelMode.TRAIN,
                                                                        save=save_train,
                                                                        augment_labeled_data=False,
                                                                        augment_unlabeled_data=False,
                                                                        train_adv=train_adv)

            with torch.no_grad():
                val_dice, val_batch_dice = self._evaluate_loop(val_dataloader=self.dataloaders['val'],
                                                               epoch=epoch, mode=ModelMode.EVAL,
                                                               save=save_val)
            self.schedulerStep()

            for k in metrics:
                assert metrics[k][epoch].shape == eval(k).shape, (metrics[k][epoch].shape, eval(k).shape)
                metrics[k][epoch] = eval(k)
            for k, e in metrics.items():
                np.save(Path(self.save_dir, f"{k}.npy"), e.detach().cpu().numpy())

            df = pd.DataFrame(
                {
                    **{f"train_dice_{i}": metrics["train_dice"].mean(1)[:, 0, i].cpu() for i in self.axises},
                    **{f"train_unlab_dice_{i}": metrics["train_unlab_dice"].mean(1)[:, 0, i].cpu() for i in
                       self.axises},
                    **{f"val_dice_{i}": metrics["val_dice"].mean(1)[:, 0, i].cpu() for i in self.axises},
                    # using the axis = 3
                    **{f"val_batch_dice_{i}": metrics["val_batch_dice"].mean(1)[:, 0, i].cpu() for i in self.axises}
                })

            df.to_csv(Path(self.save_dir, self.metricname), float_format="%.4f", index_label="epoch")

            current_metric = val_dice[:, 0, self.axises].mean()
            self.checkpoint(current_metric, epoch)

    def _train_loop(self, labeled_dataloader: DataLoader, unlabeled_dataloader: DataLoader, epoch: int,
                    mode: ModelMode, save: bool, augment_labeled_data=False, augment_unlabeled_data=False,
                    train_adv=False):
        self.segmentator.set_mode(mode)

        labeled_dataloader.dataset.set_mode(
            ModelMode.TRAIN if mode == ModelMode.TRAIN and augment_labeled_data else ModelMode.EVAL)
        unlabeled_dataloader.dataset.training = ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        assert self.segmentator.training == True
        assert labeled_dataloader.dataset.training == ModelMode.TRAIN if mode == ModelMode.TRAIN and augment_labeled_data \
            else ModelMode.EVAL
        assert unlabeled_dataloader.dataset.training == ModelMode.TRAIN if mode == ModelMode.TRAIN and augment_unlabeled_data \
            else ModelMode.EVAL

        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"

        # Here the concept of epoch is defined as the epoch
        n_batch = labeled_dataloader.__len__()
        n_img = n_batch * labeled_dataloader.batch_size if labeled_dataloader.drop_last else len(
            labeled_dataloader.dataset)
        # S labeled dataset + 1 unlabeled dataset
        n_unlab_img = n_batch * unlabeled_dataloader.batch_size if unlabeled_dataloader.drop_last else len(
            unlabeled_dataloader.dataset)

        coef_dice = torch.zeros(n_img, 1, self.C)
        unlabel_coef_dice = torch.zeros(n_unlab_img, 1, self.C)
        loss_log = torch.zeros(n_batch, 1)

        lab_done = 0
        unlab_done = 0

        ## build fake_iterator
        fake_labeled_iterator = iterator_(dcopy(labeled_dataloader))

        fake_unlabeled_iterator = iterator_(dcopy(unlabeled_dataloader))

        n_batch_iter = tqdm_(range(n_batch))

        report_iterator = iterator_(['label', 'unlab'])
        report_status = 'label'
        lab_dsc_dict: dict
        lab_mean_dict: dict

        for batch_num in n_batch_iter:
            if batch_num % 30 == 0 and train_adv:
                report_status = report_iterator.__next__()

            [[img, gt], _, path] = fake_labeled_iterator.__next__()
            img, gt = img.to(self.device), gt.to(self.device)
            lab_B = img.shape[0]
            ## backward and update when the mode = ModelMode.TRAIN
            pred, sup_loss = self.segmentator.update(img, gt, criterion=self.criterion,
                                                     mode=ModelMode.TRAIN)
            c_dice = dice_coef(*self.toOneHot(pred, gt))  # shape: B, axises

            if save:
                save_images(pred2class(pred), names=path, root=self.save_dir, mode='train', iter=epoch)

            batch_slice = slice(lab_done, lab_done + lab_B)
            ## record supervised data
            coef_dice[batch_slice] = c_dice.unsqueeze(1)
            loss_log[batch_num] = sup_loss
            lab_done += lab_B

            if train_adv:
                [[unlab_img, unlab_gt], _, path] = fake_unlabeled_iterator.__next__()
                unlab_B = unlab_img.shape[0]
                batch_slice = slice(unlab_done, unlab_done + unlab_B)
                unlab_img, unlab_gt = unlab_img.to(self.device), unlab_gt.to(self.device)
                unlab_img_adv = VATGenerator(self.segmentator.torchnet, eplision=10)(dcopy(unlab_img))
                assert unlab_img.shape == unlab_img_adv.shape
                adv_pred = self.segmentator.predict(unlab_img_adv, logit=False)
                real_pred = self.segmentator.predict(unlab_img, logit=False)
                unlab_dice = dice_coef(*self.toOneHot(real_pred, unlab_gt))
                unlabel_coef_dice[batch_slice] = unlab_dice.unsqueeze(1)
                unlab_done += unlab_B
                if save:
                    save_images(pred2class(real_pred), names=path, root=self.save_dir, mode='unlab', iter=epoch)

                adv_loss = KL_Divergence_2D(reduce=True)(adv_pred, real_pred.detach()) * self.adv_scheduler.value

                self.segmentator.optimizer.zero_grad()
                adv_loss.backward()
                self.segmentator.optimizer.step()

            lab_big_slice = slice(0, lab_done)
            unlab_big_slice = slice(0, unlab_done)

            lab_dsc_dict = {f"DSC{n}": coef_dice[lab_big_slice, 0, n].mean() for n in self.axises}

            lab_mean_dict = {"DSC": coef_dice[lab_big_slice, 0, self.axises].mean()}

            unlab_dsc_dict = {f"DSC{n}": unlabel_coef_dice[unlab_big_slice, 0, n].mean() for n in self.axises}

            unlab_mean_dict = {"DSC": unlabel_coef_dice[unlab_big_slice, 0, self.axises].mean()}

            stat_dict = {**lab_dsc_dict, **lab_mean_dict} if report_status == 'label' \
                else {**lab_dsc_dict, **lab_mean_dict, **unlab_dsc_dict, **unlab_mean_dict}

            # to delete null dicts
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}

            n_batch_iter.set_postfix(nice_dict)
            n_batch_iter.set_description(f'{report_status}->> loss:{loss_log[:batch_num].mean().item():.3f}')

        ## make sure that the dicts are for the labeled dataset

        stat_dict = {**lab_dsc_dict, **lab_mean_dict}
        nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}
        print(
            f"{desc} " + ', '.join([f'{k}:{float(v):.3f}' for k, v in nice_dict.items()])
        )
        return coef_dice, unlabel_coef_dice, loss_log

    def _evaluate_loop(self, val_dataloader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL,
                       save: bool = True):
        self.segmentator.set_mode(mode)
        val_dataloader.dataset.set_mode(ModelMode.EVAL)
        assert self.segmentator.training == False
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        n_batch = len(val_dataloader)
        n_img = val_dataloader.batch_size * n_batch if val_dataloader.drop_last == True else len(val_dataloader.dataset)
        coef_dice = torch.zeros(n_img, 1, self.C)
        batch_dice = torch.zeros(n_batch, 1, self.C)
        val_dataloader = tqdm_(val_dataloader)
        done = 0

        nice_dict: dict

        for batch_num, [(img, gt), _, path] in enumerate(val_dataloader):
            img, gt = img.to(self.device), gt.to(self.device)
            B = img.shape[0]
            preds = self.segmentator.predict(img, logit=True)
            c_dices: Tensor = dice_coef(*self.toOneHot(preds, gt))  # shape: B, axises
            b_dices: Tensor = dice_batch(*self.toOneHot(preds, gt))
            batch_slice = slice(done, done + B)
            coef_dice[batch_slice] = c_dices.unsqueeze(1)
            batch_dice[batch_num] = b_dices.unsqueeze(0)
            done += B

            if save:
                save_images(pred2class(preds), names=path, root=self.save_dir, mode='eval',
                            iter=epoch)

            big_slice = slice(0, done)

            dsc_dict = {f"DSC{n}": coef_dice[big_slice, 0, n].mean() for n in self.axises}
            mean_dict = {"DSC": coef_dice[big_slice, 0, self.axises].mean()}

            bdsc_dict = {f"bDSC{n}": batch_dice[:batch_num + 1, 0, n].mean() for n in self.axises}

            stat_dict = {**dsc_dict, **mean_dict, **bdsc_dict}

            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}

            val_dataloader.set_postfix(nice_dict)

        print(
            f"{desc} " + ', '.join([f'{k}:{float(v):.3f}' for k, v in nice_dict.items()])
        )
        return coef_dice, batch_dice

    def schedulerStep(self):
        super().schedulerStep()
        self.adv_scheduler.step()
