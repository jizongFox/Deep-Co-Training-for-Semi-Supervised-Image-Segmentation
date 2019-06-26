import unittest
from unittest import TestCase
from generalframework.metrics.metric import ListAggregatedMeter, AggragatedMeter
from generalframework.metrics import DiceMeter
import torch


class TestListAggregatedMeter(TestCase):
    def setUp(self):
        self.aggrated_meters = [AggragatedMeter(DiceMeter(method='2d', C=4)) for _ in range(3)]
        self.list_aggrated_meters = ListAggregatedMeter(self.aggrated_meters, names=['1', '2', '3'])
        self.imgs = torch.randn(10, 4, 256, 256)
        self.gts = torch.randint(low=0, high=4, size=(10, 256, 256))
        self.max_epochs = 3

    def run_epochs(self):
        for epoch in range(self.max_epochs):
            for i in range(5):
                '''
                for each epoch, we add images for 5 times
                '''
                for j in range(3):
                    self.list_aggrated_meters[j].add(self.imgs, self.gts.squeeze(1))
            self.list_aggrated_meters.step()

    def test_add_function(self):
        self.run_epochs()
        for i in range(len(self.aggrated_meters)):
            self.assertAlmostEqual(self.list_aggrated_meters.ListAggragatedMeter[i].summary().shape[0], self.max_epochs)

    def test_summary(self):
        self.run_epochs()
        summary = self.list_aggrated_meters.summary()
        print()
        print('summary of the list_avg_meters:\n', summary)
        detailed_summary = self.list_aggrated_meters.detailed_summary()
        print()
        print('detailed summary of the list_avg_meters:\n', detailed_summary)
        self.assertAlmostEqual(summary.shape[0], self.max_epochs)
        self.assertAlmostEqual(detailed_summary.shape[0], self.max_epochs)
