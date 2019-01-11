from abc import ABC, abstractmethod
from generalframework.arch import get_arch
from typing import Union, Callable, List
import torch
from torch.nn import functional as F
from torch import Tensor
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import NLLLoss
from generalframework.utils import dice_coef, probs2one_hot, class2one_hot
from generalframework import ModelMode


class Segmentator(ABC):

    def __init__(self, arch_dict: dict, optim_dict: dict, scheduler_dict: dict) -> None:
        super().__init__()
        self.arch_dict = arch_dict
        self.optim_dict = optim_dict
        self.scheduler_dict = scheduler_dict
        self.torchnet, self.optimizer, self.scheduler = self.__setup()

    def __setup(self):
        self.arch_name = self.arch_dict['name']
        self.arch_params = {k: v for k, v in self.arch_dict.items() if k != 'name'}
        self.optim_name = self.optim_dict['name']
        self.optim_params = {k: v for k, v in self.optim_dict.items() if k != 'name'}
        self.scheduler_name = self.scheduler_dict['name']
        self.scheduler_params = {k: v for k, v in self.scheduler_dict.items() if k != 'name'}
        torchnet = get_arch(self.arch_name, self.arch_params)
        optimizer: optim.adam.Adam = getattr(optim, self.optim_name)(torchnet.parameters(), **self.optim_params)
        scheduler: lr_scheduler._LRScheduler = getattr(lr_scheduler, self.scheduler_name)(optimizer,
                                                                                          **self.scheduler_params)

        return torchnet, optimizer, scheduler

    def predict(self, img: Tensor, logit=True) -> Tensor:
        pred_logit = self.torchnet(img)
        if logit:
            return pred_logit
        return F.softmax(img, 1)

    @property
    def training(self):
        return self.torchnet.training

    def update(self, img: Tensor, gt: Tensor, criterion: NLLLoss,
               mode=ModelMode.TRAIN) -> List[
        Tensor]:
        if mode == ModelMode.TRAIN:
            assert self.training == True
        else:
            assert self.training == False

        if mode == ModelMode.TRAIN:
            self.optimizer.zero_grad()
        pred = self.predict(img)
        loss = criterion(pred, gt.squeeze(1))
        if mode == ModelMode.TRAIN:
            loss.backward()
            self.optimizer.step()

        return [pred, loss.item()]

    def schedulerStep(self):
        self.scheduler.step()

    @property
    def state_dict(self):
        return {'arch_dict': self.arch_dict, 'optim_dict': self.optim_dict, 'scheduler_dict': self.scheduler_dict,
                'net_state_dict': self.torchnet.state_dict(), 'optim_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.torchnet.load_state_dict(state_dict['net_state_dict'])
        self.optimizer.load_state_dict(state_dict['optim_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    def to(self, device: torch.device):
        self.torchnet.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL)
        if mode == ModelMode.TRAIN:
            self.train()
        elif mode == ModelMode.EVAL:
            self.eval()

    def eval(self):
        self.torchnet.eval()

    def train(self):
        self.torchnet.train()
