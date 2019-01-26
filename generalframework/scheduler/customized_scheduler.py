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
        return self.value


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

# def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
#     max_val = max_val * (float(n_labeled) / n_samples)
#     return ramp_up(epoch, max_epochs, max_val, mult)

#
# def adjust_multipliers(lambda_cot_max, lambda_diff_max, ramp_up_mult, n_labeled, n_samples, epoch, epoch_max_ramp):
#     # this is the ramp_up function for lambda_cot and lambda_diff weights on the unsupervised terms.
#     lambda_cot = weight_schedule(epoch, epoch_max_ramp, lambda_cot_max, ramp_up_mult, n_labeled, n_samples)
#     lambda_diff = weight_schedule(epoch, epoch_max_ramp, lambda_diff_max, ramp_up_mult, n_labeled, n_samples)
#     return lambda_cot, lambda_diff
#
#
# def adjust_multipliers(self, lambda_cot_max, lambda_diff_max, ramp_up_mult, epoch, epoch_max_ramp):
#     n_labeled = self.labeled_dataloaders[0].__len__()
#     n_samples = n_labeled + self.unlabeled_dataloade.__len__()
#     # this is the ramp_up function for lambda_cot and lambda_diff weights on the unsupervised terms.
#     lambda_cot = weight_schedule(epoch, self.epoch_max_ramp, self.lambda_cot_max, self.ramp_up_mult,
#                                  n_labeled, n_samples)
#     lambda_adv = weight_schedule(epoch, self.epoch_max_ramp, self.lambda_adv_max, self.ramp_up_mult,
#                                  n_labeled, n_samples)
#     # turn it into a usable pytorch object
#     lambda_cot = torch.FloatTensor([lambda_cot]).to(self.device)
#     lambda_adv = torch.FloatTensor([lambda_adv]).to(self.device)
#     return lambda_cot, lambda_adv
