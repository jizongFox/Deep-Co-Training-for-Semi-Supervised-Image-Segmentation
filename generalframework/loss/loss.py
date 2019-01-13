import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Any
from ..utils.utils import simplex
from functools import reduce


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

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
        e = input * (input + 1e-10).log()
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
