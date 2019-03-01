from typing import List
import pandas as pd

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

    def summary(self) -> dict:
        raise NotImplementedError

    def detailed_summary(self) -> dict:
        raise NotImplementedError


class AggragatedMeter(object):
    '''
    Aggragate historical information in a List.
    '''

    def __init__(self, meter: Metric = None) -> None:
        super().__init__()
        self.epoch = 0
        assert meter is not None, meter
        self.meter = meter
        self.record: List[dict] = []
        self.detailed_record: List[dict] = []

    def Step(self):
        self.epoch += 1
        summary = self.__Summary()
        self.record.append(summary)
        detailed_summary = self.__Detailed_Summary()
        self.detailed_record.append(detailed_summary)
        self.meter.reset()

    def __Summary(self):
        return self.meter.summary()

    def __Detailed_Summary(self):
        return self.meter.detailed_summary()

    def __repr__(self):
        return str(self.Detailed_Summary())

    def Summary(self):
        return pd.DataFrame(self.record)

    def Detailed_Summary(self):
        return pd.DataFrame(self.detailed_record)

    def Add(self, *input):
        self.meter.add(*input)

    def Reset(self):
        self.meter.reset()
        self.record = []
        self.detailed_record = []

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


class AggregatedDict(object):
    pass
