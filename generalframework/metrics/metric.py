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

    def summary(self):
        raise NotImplementedError


class AggragatedMeter(object):
    '''
    Aggragate historical information in a List.
    '''
    def __init__(self, meter: Metric = None, save_dir=None) -> None:
        super().__init__()
        self.epoch = 0
        assert meter is not None,meter
        self.meter = meter
        self.record = []
        self.save_dir = save_dir

    def Step(self):
        self.epoch += 1
        instance_data = self.Summary()
        self.record.append(instance_data)
        self.meter.reset()

    def Summary(self):
        return self.meter.summary()

    def Add(self,*input):
        self.meter.add(*input)

    def Reset(self):
        self.meter.reset()
        self.record=[]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'meter'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)