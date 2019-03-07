from typing import Dict
from torch import nn
import pandas as pd
from generalframework import ModelMode

from .trainer import Trainer
from ..loss import KL_Divergence_2D
from ..metrics import DiceMeter, AverageValueMeter
from ..models import Segmentator
from ..utils import *
from ..utils.AEGenerator import VATGenerator
from ..scheduler import *

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VatTrainer(Trainer):
    def __init__(self, segmentator: Segmentator, dataloaders: Dict[str, DataLoader], criterion: nn.Module,
                 max_epoch: int = 100,
                 save_dir: str = 'tmp',
                 device: str = 'cpu',
                 axises: List[int] = [1, 2, 3],
                 checkpoint: str = None,
                 metricname: str = 'metrics.csv',
                 whole_config=None,
                 adv_scheduler_dict: dict = None) -> None:
        super().__init__(segmentator=segmentator, dataloaders=dataloaders, criterion=criterion,
                         max_epoch=max_epoch, save_dir=save_dir,
                         device=device, axises=axises, checkpoint=checkpoint, metricname=metricname,
                         whole_config=whole_config)
        self.adv_scheduler = eval(adv_scheduler_dict['name'])(
            **{k: v for k, v in adv_scheduler_dict.items() if k != 'name'})

    def start_training(self, train_adv=False, adv_training_dict: dict = {}, save_train=False, save_val=False,
                       use_tqdm=True):
        n_class: int = self.C

        metrics = {"val_dice": torch.zeros((self.max_epoch, 1, n_class, 2), device=self.device).type(torch.float32),
                   "val_batch_dice": torch.zeros((self.max_epoch, 1, n_class, 2), device=self.device).type(
                       torch.float32),
                   "train_dice": torch.zeros((self.max_epoch, 1, n_class, 2), device=self.device).type(
                       torch.float32),
                   "train_unlab_dice": torch.zeros((self.max_epoch, 1, n_class, 2), device=self.device).type(
                       torch.float32),
                   "train_loss": torch.zeros((self.max_epoch), device=self.device).type(torch.float32),
                   "adv_loss": torch.zeros((self.max_epoch), device=self.device).type(torch.float32)}

        for epoch in range(self.start_epoch, self.max_epoch):

            train_dice, train_unlab_dice, train_loss, adv_loss = self._train_loop(
                labeled_dataloader=self.dataloaders['lab'],
                unlabeled_dataloader=self.dataloaders['unlab'],
                epoch=epoch, mode=ModelMode.TRAIN,
                save=save_train,
                augment_labeled_data=False,
                augment_unlabeled_data=False,
                train_adv=train_adv,
                use_tqdm=use_tqdm,
                **adv_training_dict)

            with torch.no_grad():
                val_dice, val_batch_dice = self._evaluate_loop(val_dataloader=self.dataloaders['val'],
                                                               epoch=epoch, mode=ModelMode.EVAL,
                                                               save=save_val,
                                                               use_tqdm=use_tqdm)
            self.schedulerStep()

            for k in metrics.keys():
                assert metrics[k][epoch].shape == eval(k).shape, (metrics[k][epoch].shape, eval(k).shape)
                metrics[k][epoch] = eval(k)
            for k, e in metrics.items():
                np.save(Path(self.save_dir, f"{k}.npy"), e.detach().cpu().numpy())

            df = pd.DataFrame(
                {
                    **{f"train_dice_{i}": metrics["train_dice"][:, 0, i, 0].cpu() for i in self.axises},
                    **{f"train_unlab_dice_{i}": metrics["train_unlab_dice"][:, 0, i, 0].cpu() for i in
                       self.axises},
                    **{f"val_dice_{i}": metrics["val_dice"][:, 0, i, 0].cpu() for i in self.axises},
                    **{f"val_batch_dice_{i}": metrics["val_batch_dice"][:, 0, i, 0].cpu() for i in self.axises}
                })

            df.to_csv(Path(self.save_dir, self.metricname), float_format="%.4f", index_label="epoch")

            current_metric = val_dice[0, self.axises, 0].mean()
            self.checkpoint(current_metric, epoch)

    def _train_loop(self,
                    labeled_dataloader: DataLoader,
                    unlabeled_dataloader: DataLoader,
                    epoch: int,
                    mode: ModelMode,
                    save: bool,
                    augment_labeled_data=False,
                    augment_unlabeled_data=False,
                    train_adv=False,
                    vat_axises: List[int] = [1, 2, 3],
                    vat_lossname: str = 'kl',
                    eplision=0.05,
                    ip=1,
                    use_tqdm=True):
        fix_seed(epoch)
        self.segmentator.set_mode(mode)
        labeled_dataloader.dataset.set_mode(ModelMode.TRAIN if mode == ModelMode.TRAIN and augment_labeled_data \
                                                else ModelMode.EVAL)
        unlabeled_dataloader.dataset.training = ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        assert self.segmentator.training
        assert labeled_dataloader.dataset.training == ModelMode.TRAIN \
            if mode == ModelMode.TRAIN and augment_labeled_data else \
            ModelMode.EVAL
        assert unlabeled_dataloader.dataset.training == ModelMode.TRAIN \
            if mode == ModelMode.TRAIN and augment_unlabeled_data else \
            ModelMode.EVAL
        desc = f">>   Training   ({epoch})"
        # Here the concept of epoch is defined as the epoch
        n_batch = labeled_dataloader.__len__()
        supdiceMeter = DiceMeter(report_axises=[0, 1, 2, 3], method='2d', C=self.C)
        advdiceMeter = DiceMeter(report_axises=[0, 1, 2, 3], method='2d', C=self.C)
        suplossMeter = AverageValueMeter()
        advlossMeter = AverageValueMeter()
        n_batch_iter = tqdm_(range(n_batch)) if use_tqdm else range(n_batch)
        fake_labeled_iterator = iterator_(dcopy(labeled_dataloader))
        fake_unlabeled_iterator = iterator_(dcopy(unlabeled_dataloader))

        for _ in n_batch_iter:
            [[img, gt], _, path] = fake_labeled_iterator.__next__()
            img, gt = img.to(self.device), gt.to(self.device)
            pred = self.segmentator.predict(img, logit=True)
            sup_loss = self.criterion(pred, gt.squeeze(1))
            supdiceMeter.add(pred, gt)
            suplossMeter.add(sup_loss.detach().cpu())

            if save:
                save_images(pred2class(pred), names=path, root=self.save_dir, mode='train', iter=epoch)
            adv_loss = 0
            if train_adv and self.adv_scheduler.value > 0:
                [[unlab_img, unlab_gt], _, path] = fake_unlabeled_iterator.__next__()
                unlab_img, unlab_gt = unlab_img.to(self.device), unlab_gt.to(self.device)
                unlab_img_adv, noise = VATGenerator(self.segmentator.torchnet, ip=ip, eplision=eplision,
                                                    axises=vat_axises) \
                    (dcopy(unlab_img), loss_name=vat_lossname)

                assert unlab_img.shape == unlab_img_adv.shape
                real_pred = self.segmentator.predict(unlab_img, logit=False)
                adv_pred = self.segmentator.predict(unlab_img_adv, logit=False)
                advdiceMeter.add(real_pred, gt)

                if save:
                    save_images(pred2class(real_pred), names=path, root=self.save_dir, mode='unlab', iter=epoch)

                adv_loss = KL_Divergence_2D(reduce=True)(p_prob=adv_pred, y_prob=real_pred.detach())
                advlossMeter.add(adv_loss.detach().cpu())

            totalloss = sup_loss + self.adv_scheduler.value * adv_loss
            self.segmentator.optimizer.zero_grad()
            totalloss.backward()
            self.segmentator.optimizer.step()
            # For recording
            lab_dsc_dict = {f"DSC{n}": supdiceMeter.value()[1][0][n].item() for n in self.axises}
            adv_dsc_dict = {f"aDSC{n}": advdiceMeter.value()[1][0][n].item() for n in self.axises}
            loss_dict = {f'sloss': suplossMeter.value()[0], f'aloss': advlossMeter.value()[0]}
            stat_dict = {
                **lab_dsc_dict,
                **adv_dsc_dict,
                **loss_dict
            }
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0 and v != float(np.nan)}
            if use_tqdm:
                n_batch_iter.set_postfix(nice_dict)
        print(
            f"{desc} " + ', '.join([f'{k}:{float(v):.3f}' for k, v in nice_dict.items()])
        )
        return torch.stack(supdiceMeter.value()[1], dim=1).unsqueeze(0), torch.stack(advdiceMeter.value()[1],
                                                                                     dim=1).unsqueeze(0), \
               suplossMeter.value()[0], advlossMeter.value()[
                   0] if train_adv and self.adv_scheduler.value > 0 else torch.tensor(0)

    def _evaluate_loop(self, val_dataloader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL,
                       save: bool = True, use_tqdm=True):
        self.segmentator.set_mode(mode)
        val_dataloader.dataset.set_mode(ModelMode.EVAL)
        assert self.segmentator.training is False
        assert val_dataloader.dataset.training == ModelMode.EVAL
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        val_dataloader = tqdm_(val_dataloader) if use_tqdm else val_dataloader
        valdiceMeter = DiceMeter(report_axises=self.axises, method='2d', C=self.C)
        valbdiceMeter = DiceMeter(report_axises=self.axises, method='3d', C=self.C)

        for batch_num, [(img, gt), _, path] in enumerate(val_dataloader):
            img, gt = img.to(self.device), gt.to(self.device)
            preds = self.segmentator.predict(img, logit=True)
            valdiceMeter.add(preds, gt)
            valbdiceMeter.add(preds, gt)
            if save:
                save_images(pred2class(preds), names=path, root=self.save_dir, mode='eval',
                            iter=epoch)

            dsc_dict = {f"DSC{n}": valdiceMeter.value()[1][0][n].item() for n in self.axises}
            bdsc_dict = {f"bDSC{n}": valbdiceMeter.value()[1][0][n].item() for n in self.axises}

            mean_dict = {"DSC": valdiceMeter.value()[0][0]}
            bmean_dict = {"bDSC": valbdiceMeter.value()[0][0]}

            stat_dict = {**dsc_dict, **mean_dict, **bdsc_dict}

            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}
            if use_tqdm:

                val_dataloader.set_postfix(nice_dict)

        print(
            f"{desc} " + ', '.join([f'{k}:{float(v):.3f}' for k, v in nice_dict.items()])
        )
        return torch.stack(valdiceMeter.value()[1], dim=1).unsqueeze(0), torch.stack(valbdiceMeter.value()[1],
                                                                                     dim=1).unsqueeze(0)

    def schedulerStep(self):
        super().schedulerStep()
        if getattr(self, "adv_scheduler"):
            self.adv_scheduler.step()
