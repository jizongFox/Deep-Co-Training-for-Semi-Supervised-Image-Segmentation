# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class, return_dict=True):
    assert label_trues.shape == label_preds.shape
    b, h, w = label_preds.shape

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    if not return_dict:
        return hist
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added # 横着加
    mean_iu = np.nanmean(iu[valid]) ## gt 出现过的mean_iu
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # cls_iu = dict(zip(range(n_class), iu))
    cls_iu = iu
    return {
        "Overall_Acc": acc,
        "Mean_Acc": acc_cls,
        "FreqW_Acc": fwavacc,
        "Validated_Mean_IoU": mean_iu,
        "Mean_IoU": np.nanmean(iu),
        "Class_IoU": torch.from_numpy(cls_iu).float(),
    }
