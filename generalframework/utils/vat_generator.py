import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models import Segmentator


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATGenerator(object):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT generator based on https://arxiv.org/abs/1704.03976
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def generate_adversarial(self, model: Segmentator, img):
        with torch.no_grad():
            pred = model.predict(img, logit=False)

        # prepare random unit tensor
        d = torch.rand(img.shape).sub(0.5).to(img.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model.torchnet):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad()
                pred_hat = model.predict(img + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_dist = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_dist.backward()
                d = _l2_normalize(d.grad)
                model.torchnet.zero_grad()

            r_adv = d * self.eps
            pred_hat = model.predict(img + r_adv)

        return pred_hat

