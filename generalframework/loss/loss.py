import torch
import torch.nn as nn
import torch.nn.functional as F


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


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()
        self.loss = F.kl_div()  # nn.KLDivLoss()

    def forward(self, predictions):
        # List of predictions form n models
        predictions = torch.cat(predictions, dim=0)
        model_probs = F.softmax(predictions, dim=2)
        # n: number of distributions, b: batch size, c: number of classes
        # (h, w): image dimensions
        n, b, c, h, w = model_probs.shape
        mixture_dist = model_probs.mean(0, keepdim=True).expand(n, b, c, h, w)
        entropy_mixture = model_probs.log()

        return torch.sum(self.loss(entropy_mixture, mixture_dist))
