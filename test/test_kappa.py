import torch

from generalframework.metrics import KappaMetrics

kappa_Meter = KappaMetrics()

for i in range(100):
    predict1 = torch.randn(2, 4, 256, 256)
    predict2 = torch.randn(2, 4, 256, 256)
    # gt = torch.randint(0, 4, size=(2, 256, 256))
    gt = predict1.max(1)[1]
    kappa_Meter.add(predicts=[predict1.max(1)[1], predict2.max(1)[1]], target=gt, considered_classes=[1, 2, 3])

print(kappa_Meter.value())
