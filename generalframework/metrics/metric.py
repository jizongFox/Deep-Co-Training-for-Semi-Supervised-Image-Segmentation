import numpy as np
import pandas as pd
from torch import Tensor
from ..utils import save_images
class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self, **kwargs):
        pass

    def value(self, **kwargs):
        pass

    def summary_epoch(self):
        pass

    def save_images(self, **kwargs):
        pass


class AggragatedMeter(object):
    '''
    Aggragate historical information in a List.
    '''

    def __init__(self, meter: Metric = None, save_dir=None) -> None:
        super().__init__()
        self.epoch = 0
        assert meter is not None
        self.meter = meter
        self.record = []
        self.save_dir = save_dir

    def Step(self):
        self.epoch += 1
        self.Summary()
        self.meter.reset()

    def Summary(self):
        self.record.append(self.meter.summary_epoch())
        return pd.DataFrame(self.record)

    def Add(self,*input):
        self.meter.add(*input)

    def Save_Images(self,pred_logit:Tensor, filenames, mode='train',seg_num=None):
        assert pred_logit.shape.__len__()==4
        pred_mask = pred_logit.max(1)[1]
        save_images(pred_mask,filenames, root=self.save_dir, mode=mode,iter=self.epoch, seg_num=seg_num)

class Horizontal_Meter(object):

    def __init__(self,*aggregatedMeters) -> None:
        super().__init__()
        self.meters = aggregatedMeters

    def Step(self):
        [m.step for m in self.meters]



class AverageValueMeter(Metric):
    def __init__(self, name='Average Meter'):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0
        self.name = name

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
