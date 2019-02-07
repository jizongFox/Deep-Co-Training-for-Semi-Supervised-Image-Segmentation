import torch
import numpy as np


class Scheduler(object):
    def __init__(self, last_epoch=-1):
        self.last_epoch = last_epoch

    def get_current_value(self):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    @property
    def value(self):
        # return self.value
        return NotImplementedError


class RampScheduler(Scheduler):

    def __init__(self, max_epoch, max_value, ramp_mult, last_epoch=-1):
        super().__init__(last_epoch)
        self.max_epoch = max_epoch
        self.max_value = max_value
        self.mult = ramp_mult
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.ramp_up(self.epoch, self.max_epoch, self.max_value, self.mult)

    @staticmethod
    def ramp_up(epoch, max_epochs, max_val, mult):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


class ConstantScheduler(Scheduler):

    def __init__(self, max_value=1.0, last_epoch=-1):
        super().__init__(last_epoch)
        self.max_value = max_value
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.max_value


class RampDownScheduler(RampScheduler):

    def __init__(self, max_epoch, max_value, ramp_mult, min_val, cutoff, last_epoch=-1):
        super().__init__(last_epoch)
        self.max_epoch = max_epoch
        self.max_value = max_value
        self.mult = ramp_mult
        self.epoch = 0
        self.min_val = min_val
        self.cutoff = cutoff

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.ramp_down(self.epoch, self.max_epoch, self.max_value, self.mult, self.min_val, self.cutoff)

    @staticmethod
    def ramp_down(epoch, max_epochs, max_val, mult, min_val, cutoff):
        assert cutoff < max_epochs
        if epoch == 0:
            return 1.
        elif epoch >= cutoff:
            return min_val
        return 1 - max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
