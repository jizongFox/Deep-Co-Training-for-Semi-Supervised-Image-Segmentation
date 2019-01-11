from abc import ABC, abstractmethod
from generalframework import LOGGER, config_logger
from generalframework.utils import *
from torch.utils.data import DataLoader
import torch, os, shutil, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from ..models import Segmentator
from typing import Dict, Callable, List, Union
import warnings
from generalframework import ModelMode
import shutil


class Base(ABC):

    @abstractmethod
    def start_training(self):
        raise NotImplementedError

    @abstractmethod
    def _main_loop(self, dataloader, epoch, mode, save, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, **kwargs):
        raise NotImplementedError


class Trainer(Base):
    def __init__(self, segmentator: Segmentator, dataloaders: Dict[str, DataLoader], criterion: nn.Module,
                 max_epoch: int = 100,
                 save_dir: str = 'tmp', save_train=False, save_val=False,
                 device: str = 'cpu', axises: List[int] = [0],
                 checkpoint: str = None, metricname: str = 'metrics.csv') -> None:
        super().__init__()
        self.max_epoch = max_epoch
        self.segmentator = segmentator
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_train = save_train
        self.save_val = save_val
        self.device = torch.device(device)
        self.C = segmentator.arch_params['num_classes']
        self.axises = axises
        self.best_score = -1
        self.start_epoch = 0
        self.metricname = metricname

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        self.to(self.device)

    def _load_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        assert checkpoint.exists(), checkpoint
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.segmentator.load_state_dict(state_dict['segmentator'])
        self.best_score = state_dict['best_score']
        self.start_epoch = state_dict['best_epoch'] + 1
        print(f'>>>  {checkpoint} has been loaded successfully. Best score {self.best_score} @ {self.start_epoch}.')
        self.segmentator.train()

    def to(self, device: torch.device):
        self.segmentator.to(device)
        self.criterion.to(device)

    def start_training(self):
        ## prepare for something:

        n_class: int = self.C
        train_n: int = len(self.dataloaders['train'].dataset)  # Number of images in dataset
        train_b: int = len(self.dataloaders['train'])  # Number of iteration per epoch: different if batch_size > 1
        val_n: int = len(self.dataloaders['val'].dataset)
        val_b: int = len(self.dataloaders['val'])

        metrics = {"val_dice": torch.zeros((self.max_epoch, val_n, n_class), device=device).type(torch.float32),
                   "val_batch_dice": torch.zeros((self.max_epoch, val_b, n_class), device=device).type(torch.float32),
                   "val_loss": torch.zeros((self.max_epoch, val_b), device=device).type(torch.float32),
                   "train_dice": torch.zeros((self.max_epoch, train_n, n_class), device=device).type(torch.float32),
                   "train_loss": torch.zeros((self.max_epoch, train_b), device=device).type(torch.float32)}

        train_loader, val_loader = self.dataloaders['train'], self.dataloaders['val']
        for epoch in range(self.start_epoch, self.max_epoch):
            train_dice, _, train_loss = self._main_loop(train_loader, epoch, mode=ModelMode.TRAIN, save=self.save_train)
            with torch.no_grad():
                val_dice, val_batch_dice, val_loss = self._main_loop(val_loader, epoch, mode=ModelMode.EVAL,
                                                                     save=self.save_val)
            self.segmentator.schedulerStep()

            for k in metrics:
                assert metrics[k][epoch].shape == eval(k).shape, (metrics[k][epoch].shape, eval(k).shape)
                metrics[k][epoch] = eval(k)
            for k, e in metrics.items():
                np.save(Path(self.save_dir, f"{k}.npy"), e.cpu().numpy())

            df = pd.DataFrame({"train_loss": metrics["train_loss"].mean(dim=1).cpu().numpy(),
                               "val_loss": metrics["val_loss"].mean(dim=1).cpu().numpy(),
                               "train_dice": metrics["train_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               # using the axis = 3
                               "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               # using the axis = 3
                               "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy()})
            ## the saved metrics are with only axis==3, as the foreground dice.

            df.to_csv(Path(self.save_dir, self.metricname), float_format="%.4f", index_label="epoch")

            current_metric = val_dice[:, self.axises].mean()
            self.checkpoint(current_metric, epoch)

    def _main_loop(self, dataloader: DataLoader, epoch: int, mode, save: bool):
        self.segmentator.set_mode(mode)
        dataloader.dataset.set_mode(mode)
        desc = f">> Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        assert dataloader.dataset.training == mode

        n_img = len(dataloader.dataset)
        n_batch = len(dataloader)
        coef_dice = torch.zeros(n_img, self.C)
        batch_dice = torch.zeros(n_batch, self.C)
        loss_log = torch.zeros(n_batch)
        dataloader = tqdm_(dataloader)
        done = 0
        for i, (imgs, metainfo, filenames) in enumerate(dataloader):
            B = filenames.__len__()
            imgs = [img.to(self.device) for img in imgs]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                [preds, loss] = self.segmentator.update(imgs[0], imgs[1], self.criterion, mode=mode)
            ohpredmask, ohmask = self.toOneHot(preds, imgs[1])
            c_dice = dice_coef(ohpredmask, ohmask)

            if mode == ModelMode.EVAL:
                b_dice = dice_batch(ohpredmask, ohmask)

            batch = slice(done, done + B)
            coef_dice[batch] = c_dice

            if mode == ModelMode.EVAL:
                batch_dice[batch] = b_dice

            loss_log[batch] = loss
            done += B

            if save:
                save_images(segs=preds.max(1)[1], names=filenames, root=self.save_dir, mode=mode.value.lower(),
                            iter=epoch)

            ## for visualization
            big_slice = slice(0, done)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": coef_dice[big_slice, n].mean() for n in self.axises}

            bdsc_dict = {f"bDSC{n}": batch_dice[big_slice, n].mean() for n in
                         self.axises}

            mean_dict = {"DSC": coef_dice[big_slice, self.axises].mean()}

            stat_dict = {**dsc_dict, **bdsc_dict, **mean_dict,
                         "loss": loss_log[:i].mean()}
            # to delete null dicts
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items() if v != 0}

            dataloader.set_postfix(nice_dict)  ## using average value of the dict

        print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

        return coef_dice, batch_dice, loss_log

    def checkpoint(self, metric, epoch, filename='best.pth'):
        if metric <= self.best_score:
            return
        else:
            self.best_score = metric
            state_dict = self.segmentator.state_dict
            state_dict = {'segmentator': state_dict, 'best_score': metric, 'best_epoch': epoch}
            save_path = Path(os.path.join(self.save_dir, filename))
            torch.save(state_dict, os.path.join(self.save_dir, filename))
            print(f'model saved @ {epoch} with metrics of {metric}')

    @classmethod
    def toOneHot(cls, pred_logit, mask):
        oh_predmask = probs2one_hot(F.softmax(pred_logit, 1))
        oh_mask = class2one_hot(mask.squeeze(1), pred_logit.shape[1])
        assert oh_predmask.shape == oh_mask.shape
        return oh_predmask, oh_mask
