import torch

from generalframework.metrics import DiceMeter
from generalframework.metrics.metric import ListAggregatedMeter, AggragatedMeter


def test_ListAggregatedMeter():
    train_2d_dice = AggragatedMeter(DiceMeter(method='2d', C=4))
    train_3d_dice = AggragatedMeter(DiceMeter(method='3d', C=4))
    pred_1 = torch.randn(10, 4, 256, 256)
    gt = torch.randint(0,4,(10,256,256))

    train_2d_dice.Add(pred_1,gt)
    train_3d_dice.Add(pred_1,gt)

    ListedAggregatedmeters = ListAggregatedMeter(
        listAggregatedMeter=
        [train_2d_dice, train_3d_dice],
        names=['2d', '3d']
    )
    print(ListAggregatedMeter[0])

if __name__ == '__main__':
    test_ListAggregatedMeter()
