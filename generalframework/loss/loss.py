import warnings
from functools import reduce
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import simplex


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        if weight is not None:
            weight = torch.Tensor(weight)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class PartialCrossEntropyLoss2d(nn.Module):

    def __init__(self, reduce=True, size_average=True):
        super(PartialCrossEntropyLoss2d, self).__init__()
        weight = torch.Tensor([0, 1])
        self.loss = nn.NLLLoss(weight=weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class MSE_2D(nn.Module):
    def __init__(self):
        super(MSE_2D, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        prob = F.softmax(input, dim=1)[:, 1].squeeze()
        target = target.squeeze()
        assert prob.shape == target.shape
        return self.loss(prob, target.float())


class Entropy_2D(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        the definition of Entropy is - \sum p(xi) log (p(xi))
        '''

    def forward(self, input: torch.Tensor):
        assert input.shape.__len__() == 4
        b, _, h, w = input.shape
        assert simplex(input)
        e = input * (input + 1e-16).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, h, w])
        return e


class JSD_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # self.C = num_probabilities
        self.entropy = Entropy_2D()

    def forward(self, input: List[torch.Tensor]):
        # assert self.C == input.__len__()
        for inprob in input:
            assert simplex(inprob, 1)
        mean_prob = reduce(lambda x, y: x + y, input) / len(input)
        f_term = self.entropy(mean_prob)
        mean_entropy = sum(list(map(lambda x,: self.entropy(x), input))) / len(input)
        assert f_term.shape == mean_entropy.shape
        return f_term - mean_entropy


class KL_Divergence_2D(nn.Module):

    def __init__(self, reduce=False, eps=1e-10):
        super().__init__()
        self.reduce = reduce
        self.eps = eps

    def forward(self, p_prob: torch.Tensor, y_prob: torch.Tensor):
        '''
        :param p_probs:
        :param y_prob: the Y_logit is like that for crossentropy
        :return: 2D map?
        '''
        assert simplex(p_prob, 1)
        assert simplex(y_prob, 1)

        logp = (p_prob + self.eps).log()
        logy = (y_prob + self.eps).log()

        ylogy = (y_prob * logy).sum(dim=1)
        ylogp = (y_prob * logp).sum(dim=1)
        if self.reduce:
            return (ylogy - ylogp).mean()
        else:
            return ylogy - ylogp

class KL_Divergence_2D_Logit(nn.Module):

    def __init__(self, reduce=False, eps=1e-10):
        super().__init__()
        self.reduce = reduce
        self.eps = eps

    def forward(self, p_logit: torch.Tensor, y_logit: torch.Tensor):
        '''
        :param p_probs:
        :param y_prob: the Y_logit is like that for crossentropy
        :return: 2D map?
        '''
        # assert simplex(p_prob, 1)
        # assert simplex(y_prob, 1)

        logp = F.log_softmax(p_logit,1)
        logy = F.log_softmax(y_logit,1)
        y_prob = F.softmax(y_logit,1)

        ylogy = (y_prob * logy).sum(dim=1)
        ylogp = (y_prob * logp).sum(dim=1)
        if self.reduce:
            return (ylogy - ylogp).mean()
        else:
            return ylogy - ylogp