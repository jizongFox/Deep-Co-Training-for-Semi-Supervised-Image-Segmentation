from typing import List, Iterable
import pandas as pd
import functools

from ..utils import export


@export
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


@export
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

    def step(self):
        self.epoch += 1
        summary: dict = self.__summary()
        self.record.append(summary)
        detailed_summary = self.__detailed_summary()
        self.detailed_record.append(detailed_summary)
        self.meter.reset()

    def __summary(self) -> dict:
        return self.meter.summary()

    def __detailed_summary(self) -> dict:
        return self.meter.detailed_summary()

    def __repr__(self):
        return str(self.detailed_summary())

    # public interface of dict
    def summary(self, if_dict=False):
        if if_dict:
            return self.record
        return pd.DataFrame(self.record)

    # public interface
    def detailed_summary(self, if_dict=False):
        if if_dict:
            return self.detailed_record
        return pd.DataFrame(self.detailed_record)

    def add(self, *input_, **kwargs):
        self.meter.add(*input_, **kwargs)

    def reset(self):
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


@export
class ListAggregatedMeter(object):

    def __init__(self,
                 listAggregatedMeter: List[AggragatedMeter],
                 names: Iterable[str] = None
                 ) -> None:
        super().__init__()
        self.ListAggragatedMeter: List[AggragatedMeter] = listAggregatedMeter
        self.names = names
        assert self.ListAggragatedMeter.__len__() == self.names.__len__()
        assert isinstance(self.ListAggragatedMeter, list), type(self.ListAggragatedMeter)

    def __getitem__(self, index: int):
        return self.ListAggragatedMeter[index]

    def step(self) -> None:
        [m.step() for m in self.ListAggragatedMeter]

    def add(self, **kwargs) -> None:
        raise NotImplementedError("use indexing and adding one by one")

    def summary(self) -> pd.DataFrame:
        '''
        summary on the list of subsummarys, merging them together.
        :return:
        '''

        def change_dataframe_name(dataframe: pd.DataFrame, name: str):
            dataframe.columns = list(map(lambda x: name + '_' + x, dataframe.columns))
            return dataframe

        list_of_summary = [change_dataframe_name(self.ListAggragatedMeter[i].summary(), n) \
                           for i, n in enumerate(self.names)]

        summary = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), list_of_summary)

        return pd.DataFrame(summary)

    def detailed_summary(self) -> pd.DataFrame:
        '''
        summary on the list of subsummarys, merging them together.
        :return:
        '''

        def change_dataframe_name(dataframe: pd.DataFrame, name: str):
            dataframe.columns = list(map(lambda x: name + '_' + x, dataframe.columns))
            return dataframe

        list_of_detailed_summary = [change_dataframe_name(self.ListAggragatedMeter[i].detailed_summary(), n) for i, n in
                                    enumerate(self.names)]

        summary = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                                   list_of_detailed_summary)

        return pd.DataFrame(summary)
