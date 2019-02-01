import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import yaml
from generalframework import ModelMode
from ..utils import *
from ..models import Segmentator
from ..utils.metrics import scores


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


class Trainer_City(Base):
    def __init__(self, segmentator: Segmentator, dataloaders: Dict[str, DataLoader], criterion: nn.Module,
                 max_epoch: int = 100,
                 save_dir: str = 'tmp',
                 device: str = 'cpu',
                 axises: List[int] = None,
                 checkpoint: str = None,
                 metricname: str = 'metrics.csv',
                 whole_config=None) -> None:
        super().__init__()
        self.max_epoch = max_epoch
        self.segmentator = segmentator
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.save_dir = Path(save_dir)
        # assert not (self.save_dir.exists() and checkpoint is None), f'>> save_dir: {self.save_dir} exits.'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if whole_config:
            with open(Path(self.save_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(whole_config, outfile, default_flow_style=True)
        self.device = torch.device(device)
        self.C = segmentator.arch_params['num_classes']
        self.axises = axises if axises is not None else list(range(self.C))
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
        self.start_epoch = state_dict['best_epoch']
        print(f'>>>  {checkpoint} has been loaded successfully. Best score {self.best_score:.3f} @ {self.start_epoch}.')
        self.segmentator.train()

    def to(self, device: torch.device):
        self.segmentator.to(device)
        self.criterion.to(device)

    def start_training(self, save_train=False, save_val=False, augment_labeled_data=False):
        # prepare for something:
        n_class: int = self.C
        train_b: int = len(self.dataloaders['train'])  # Number of iteration per epoch: different if batch_size > 1
        train_n: int = train_b * self.dataloaders['train'].batch_size \
            if self.dataloaders['train'].drop_last \
            else len(self.dataloaders['train'].dataset)
        # Number of images in dataset

        val_b: int = len(self.dataloaders['val'])
        val_n: int = val_b * self.dataloaders['val'].batch_size if self.dataloaders['val'].drop_last \
            else len(self.dataloaders['val'].dataset)

        metrics = {"val_loss": torch.zeros((self.max_epoch, val_b), device=self.device).type(torch.float32),
                   "val_mean_IoU": torch.zeros((self.max_epoch, val_b), device=self.device).type(torch.float32),
                   "val_mean_Acc": torch.zeros((self.max_epoch, val_b), device=self.device).type(torch.float32),
                   "val_class_IoU": torch.zeros((self.max_epoch, val_b, 1, self.C), device=self.device).type(
                       torch.float32),

                   "train_loss": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_mean_IoU": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_mean_Acc": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_class_IoU": torch.zeros((self.max_epoch, train_b, 1, self.C), device=self.device).type(
                       torch.float32)
                   }

        train_loader, val_loader = self.dataloaders['train'], self.dataloaders['val']
        for epoch in range(self.start_epoch, self.max_epoch):
            train_loss, _, train_mean_Acc, train_mean_IoU, _, train_class_IoU = self._main_loop(train_loader, epoch,
                                                                                                mode=ModelMode.TRAIN,
                                                                                                augment_data=augment_labeled_data,
                                                                                                save=save_train)
            with torch.no_grad():
                val_loss, _, val_mean_Acc, val_mean_IoU, _, val_class_IoU = self._main_loop(val_loader, epoch,
                                                                                            mode=ModelMode.EVAL,
                                                                                            save=save_val)
            self.schedulerStep()

            for k in metrics:
                assert metrics[k][epoch].shape == eval(k).shape, (k, metrics[k][epoch].shape, eval(k).shape)
                metrics[k][epoch] = eval(k)
            for k, e in metrics.items():
                np.save(Path(self.save_dir, f"{k}.npy"), e.detach().cpu().numpy())

            # df = pd.DataFrame(
            #     {
            #         **{f"train_dice_{i}": metrics["train_dice"].mean(1)[:, 0, i].cpu() for i in self.axises},
            #         **{f"val_dice_{i}": metrics["val_dice"].mean(1)[:, 0, i].cpu() for i in self.axises},
            #         **{f"val_batch_dice_{i}": metrics["val_batch_dice"].mean(1)[:, 0, i].cpu() for i in self.axises}
            #     })
            #
            # df.to_csv(Path(self.save_dir, self.metricname), float_format="%.4f", index_label="epoch")

            current_metric = val_mean_IoU.mean()
            self.checkpoint(current_metric, epoch)

    def _main_loop(self, dataloader: DataLoader, epoch: int, mode, augment_data: bool = False, save: bool = False):
        self.segmentator.set_mode(mode)
        dataloader.dataset.set_mode(mode)
        if augment_data is False and mode == ModelMode.TRAIN:
            dataloader.dataset.set_mode(ModelMode.EVAL)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        assert dataloader.dataset.training == mode if augment_data else ModelMode.EVAL

        n_batch = len(dataloader)

        # for dataloader with batch_sampler, there is no dataloader.batch_size
        n_img = n_batch * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)

        loss_log = torch.zeros(n_batch)
        FreqW_Acc = torch.zeros(n_batch)
        Mean_Acc = torch.zeros(n_batch)
        Mean_IoU = torch.zeros(n_batch)
        Overall_Acc = torch.zeros(n_batch)
        Class_IoU = torch.zeros(n_batch, 1, self.C)

        dataloader = tqdm_(dataloader)
        for i, (imgs, metainfo, filenames) in enumerate(dataloader):
            imgs = [img.to(self.device) for img in imgs]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                preds, loss = self.segmentator.update(imgs[0], imgs[1], self.criterion, mode=mode)
            # c_dice = dice_coef(*self.toOneHot(preds, imgs[1]))

            c_dice = scores(label_preds=preds.max(1)[1].cpu().detach().numpy(),
                            label_trues=imgs[1].squeeze(1).cpu().numpy(),
                            n_class=19)

            for k, v in c_dice.items():
                eval(k)[i] = v
            loss_log[i] = loss

            # if save:
            #     save_images(segs=preds.max(1)[1], names=filenames, root=self.save_dir, mode=mode.value.lower(),
            #                 iter=epoch)

            # for visualization

            mean_iou_dict = {"mIoU": Mean_IoU[:i + 1].mean().item()}

            mean_cls_iou_dict = {f"c{j}": Class_IoU[:i + 1, 0, j].mean().item() for j in self.axises}

            stat_dict = {**mean_iou_dict, **mean_cls_iou_dict}
            # to delete null dicts
            nice_dict = {k: f"{v:.2f}" for (k, v) in stat_dict.items() if v != 0 or v != float(np.nan)}

            dataloader.set_description(
                f'{"trals" if mode == ModelMode.TRAIN else "vallos"}:{loss_log[:i + 1].mean().item():.3f}')
            dataloader.set_postfix(nice_dict)  # using average value of the dict

        print(f"{desc} " + ', '.join(f"{k}:{v}" for (k, v) in nice_dict.items()))

        return loss_log, FreqW_Acc, Mean_Acc, Mean_IoU, Overall_Acc, Class_IoU

    def checkpoint(self, metric, epoch, filename='best.pth'):
        if metric <= self.best_score:
            return
        else:
            self.best_score = metric
            state_dict = self.segmentator.state_dict
            state_dict = {'segmentator': state_dict, 'best_score': metric, 'best_epoch': epoch}
            save_path = Path(os.path.join(self.save_dir, filename))
            torch.save(state_dict, save_path)
            if Path(self.save_dir, "iter%.3d" % epoch).exists():
                if Path(self.save_dir, Path(filename).stem).exists():
                    shutil.rmtree(Path(self.save_dir, Path(filename).stem))
                shutil.copytree(Path(self.save_dir, "iter%.3d" % epoch), Path(self.save_dir, Path(filename).stem))

    @classmethod
    def toOneHot(cls, pred_logit, mask):
        oh_predmask = probs2one_hot(F.softmax(pred_logit, 1))
        oh_mask = class2one_hot(mask.squeeze(1), pred_logit.shape[1])
        assert oh_predmask.shape == oh_mask.shape
        return oh_predmask, oh_mask

    def schedulerStep(self):
        self.segmentator.schedulerStep()
