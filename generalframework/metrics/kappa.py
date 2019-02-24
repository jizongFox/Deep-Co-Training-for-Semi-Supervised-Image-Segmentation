from sklearn.metrics import cohen_kappa_score
import numpy as np
from .metric import  Metric
from typing import Union, List, Tuple
import torch
from torch import Tensor
class KappaMetrics(Metric):
    """ SKLearnMetrics computes various classification metrics at the end of a batch.
     Unforunately, doesn't work when used with generators...."""

    def __init__(self) -> None:
        super().__init__()
        self.kappa = []


    def add(self, predicts:List[Tensor], target:Tensor, considered_classes:List[int]):
        for predict in predicts:
            assert predict.shape == target.shape
        predicts =[ predict.detach().data.cpu().numpy().ravel() for predict in predicts]
        target = target.detach().data.cpu().numpy().ravel()
        mask  = [t in considered_classes for t in target]
        predicts = [predict[mask] for predict in predicts]
        target = target[mask]
        kappa_score = [cohen_kappa_score(predict, target) for  predict in predicts]
        self.kappa.append(kappa_score)


    def reset(self):
        self.kappa=[]

    def value(self):
        return torch.Tensor(self.kappa).mean(0)

