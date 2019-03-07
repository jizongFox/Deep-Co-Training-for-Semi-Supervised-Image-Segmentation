import shutil
from abc import ABC, abstractmethod
from typing import Dict
from torch import nn
import pandas as pd
import yaml

from generalframework import ModelMode
from generalframework.metrics.iou import IoU
from ..models import Segmentator
from ..utils import *


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

        # load coco pretrained model:
        try:
            state_dict = torch.load(
                'generalframework/trainer/deeplab_init_checkpoint/deeplabv2_resnet101_COCO_init.pth',
                map_location=lambda storage, loc: storage)
            new_state_dict = {k.replace('scale.', ''): v for k, v in state_dict.items()}
            assert len(set(self.segmentator.torchnet.state_dict().keys()) & set(new_state_dict.keys())) > 0
            self.segmentator.torchnet.load_state_dict(new_state_dict, strict=False)
            print('Coco pretrained model loaded')
        except Exception as e:
            print(f'Loading coco pretrained model failed with:\n {e}')

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
        train_b: int = len(self.dataloaders['train'])  # Number of iteration per epoch: different if batch_size > 1
        val_b: int = len(self.dataloaders['val'])

        metrics = {"val_loss": torch.zeros((self.max_epoch, val_b), device=self.device).type(torch.float32),
                   "val_mean_IoU": torch.zeros((self.max_epoch), device=self.device).type(torch.float32),
                   "val_mean_Acc": torch.zeros((self.max_epoch), device=self.device).type(torch.float32),
                   "val_class_IoU": torch.zeros((self.max_epoch, self.C), device=self.device).type(
                       torch.float32),
                   "train_loss": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_mean_IoU": torch.zeros((self.max_epoch), device=self.device).type(torch.float32),
                   "train_mean_Acc": torch.zeros((self.max_epoch), device=self.device).type(torch.float32),
                   "train_class_IoU": torch.zeros((self.max_epoch, self.C), device=self.device).type(
                       torch.float32)
                   }

        train_loader, val_loader = self.dataloaders['train'], self.dataloaders['val']
        for epoch in range(self.start_epoch, self.max_epoch):
            train_mean_Acc, _, train_mean_IoU, train_class_IoU, train_loss = self._main_loop(train_loader, epoch,
                                                                                             mode=ModelMode.TRAIN,
                                                                                             augment_data=augment_labeled_data,
                                                                                             save=save_train if epoch % 10 == 0 else False)

            with torch.no_grad():
                val_mean_Acc, _, val_mean_IoU, val_class_IoU, val_loss = self._main_loop(val_loader, epoch,
                                                                                         mode=ModelMode.EVAL,
                                                                                         save=save_val if epoch % 10 == 0 else False)
            self.checkpoint(val_mean_IoU, epoch)
            self.schedulerStep()

            for k in metrics:
                assert metrics[k][epoch].shape == eval(k).shape, (k, metrics[k][epoch].shape, eval(k).shape)
                metrics[k][epoch] = eval(k)

            for k, e in metrics.items():
                np.save(Path(self.save_dir, f"{k}.npy"), e.detach().cpu().numpy())

            df = pd.DataFrame(
                {
                    **{f"train_mean_IoU": metrics["train_mean_IoU"].cpu()},
                    **{f"val_mean_IoU": metrics["val_mean_IoU"].cpu()},
                })
            df.to_csv(Path(self.save_dir, self.metricname), float_format="%.4f", index_label="epoch")

    def _main_loop(self, dataloader: DataLoader, epoch: int, mode, augment_data: bool = False, save: bool = False):
        self.segmentator.set_mode(mode)
        dataloader.dataset.set_mode(mode)
        if augment_data is False and mode == ModelMode.TRAIN:
            dataloader.dataset.set_mode(ModelMode.EVAL)
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        assert dataloader.dataset.training == mode if augment_data else ModelMode.EVAL
        n_batch = len(dataloader)

        metrics = IoU(self.C, ignore_index=255)
        loss_log = torch.zeros(n_batch)
        mean_iou_dict: dict

        dataloader = tqdm_(dataloader)
        c_dice: dict
        for i, (imgs, metainfo, filenames) in enumerate(dataloader):

            imgs = [img.to(self.device) for img in imgs]

            preds = self.segmentator.torchnet(imgs[0])
            loss = self.criterion(preds, imgs[1].squeeze(1))
            if mode == ModelMode.TRAIN:
                self.segmentator.optimizer.zero_grad()
                loss.backward()
                self.segmentator.optimizer.step()
            metrics.add(predicted=preds, target=imgs[1])
            c_dice = metrics.value()
            loss_log[i] = loss.detach()
            if save:
                save_images(segs=preds.max(1)[1], names=map_(lambda x: Path(x).name, filenames), root=self.save_dir,
                            mode=mode.value.lower(),
                            iter=epoch)

            mean_iou_dict = {'mIoU': c_dice['Mean_IoU']}

            mean_cls_iou_dict = {f"c{j}": c_dice['Class_IoU'][j].mean().item() for j in self.axises}

            stat_dict = {**mean_iou_dict, **mean_cls_iou_dict, **{'ls': loss_log[:i + 1].mean().item()}}
            # to delete null dicts
            nice_dict = {k: f"{v:.2f}" for (k, v) in stat_dict.items() if v != 0 or v != float(np.nan)}
            #
            dataloader.set_description(
                f'{"tls" if mode == ModelMode.TRAIN else "vlos"}:{loss_log[:i + 1].mean().item():.3f}')
            dataloader.set_postfix(nice_dict)  # using average value of the dict

        stat_dict = {**mean_iou_dict, **{'ls': loss_log.mean().item()}}
        nice_dict = {k: f"{v:.2f}" for (k, v) in stat_dict.items() if v != 0 or v != float(np.nan)}

        print(f"{desc} " + ', '.join(f"{k}:{v}" for (k, v) in nice_dict.items()))

        return c_dice["Mean_Acc"], c_dice["FreqW_Acc"], c_dice["Mean_IoU"], c_dice["Class_IoU"], loss_log

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
