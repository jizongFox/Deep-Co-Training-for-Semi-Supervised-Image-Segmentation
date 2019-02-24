import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from generalframework.dataset import get_ACDC_dataloaders
from generalframework.metrics import KappaMetrics, DiceMeter
from generalframework.models import Segmentator
from generalframework.utils import probs2one_hot, class2one_hot

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='input folder directory')
    return parser.parse_args()


with open('config/ensembling_config.yaml', 'r') as f:
    config = yaml.load(f.read())
args = get_args()

input_dir = Path(args.input_dir)
checkpoints = list(input_dir.glob('best*.pth'))

dataloaders = get_ACDC_dataloaders(config['Dataset'], config['Dataloader'], quite=True)
dataloaders['val'].dataset.training = 'eval'


def load_model(checkpoint):
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))['segmentator']
    model = Segmentator(arch_dict=checkpoint['arch_dict'], optim_dict=checkpoint['optim_dict'],
                        scheduler_dict=checkpoint['scheduler_dict'])
    return model


def load_models(checkpoints):
    return [load_model(c) for c in checkpoints]


def toOneHot(pred_logit, mask):
    oh_predmask = probs2one_hot(F.softmax(pred_logit, 1))
    oh_mask = class2one_hot(mask.squeeze(1), pred_logit.shape[1])
    assert oh_predmask.shape == oh_mask.shape
    return oh_predmask, oh_mask


class Ensembleway(object):
    def __init__(self, ensembleway: str) -> None:
        super().__init__()
        assert ensembleway in ('soft', 'hard'), ensembleway
        self.ensembleway = ensembleway

    def __call__(self, predicts):
        if self.ensembleway == 'soft':
            return self._softVoting(predicts)
        else:
            return self._hardVoting(predicts)

    @staticmethod
    def _softVoting(predicts: List[Tensor]):
        assert isinstance(predicts, list), type(predicts)
        predicts_ = torch.stack(predicts, dim=0)
        predicts_ = predicts_.mean(0)
        assert predicts_.shape == predicts[0].shape
        return predicts_

    @staticmethod
    def _hardVoting(predicts: List[Tensor]):
        assert isinstance(predicts, list), type(predicts)
        hard_preds = [pred.max(1)[1] for pred in predicts]
        hard_preds = torch.cat(hard_preds, 0)
        hard_preds_ = hard_preds.detach().cpu().numpy()
        original_Shape = hard_preds_.shape
        hardvoting = lambda x: np.bincount(x).argmax()
        hard_preds_ = hard_preds_.reshape(hard_preds_.shape[0], -1)
        hard_preds = np.apply_along_axis(hardvoting, 0, hard_preds_).reshape(original_Shape[1:])
        hard_preds = torch.from_numpy(hard_preds).unsqueeze(0).float()
        hard_preds = class2one_hot(hard_preds, config['Arch']['num_classes'])
        return hard_preds.float()


device = torch.device(config['Device'])
ensemble = Ensembleway(config['Ensemble_method'])

models = load_models(checkpoints)

state_dicts = [torch.load(c, map_location=torch.device('cpu')) for c in checkpoints]

num_classes = state_dicts[0]['segmentator']['arch_dict']['num_classes']

diceMeters = [DiceMeter(method='2d', report_axises=[1, 2, 3], C=num_classes) for _ in range(checkpoints.__len__())]
bdiceMeters = [DiceMeter(method='3d', report_axises=[1, 2, 3], C=num_classes) for _ in range(checkpoints.__len__())]
ensembleMeter = DiceMeter(method='2d', report_axises=[1, 2, 3], C=num_classes)
bensembleMeter = DiceMeter(method='3d', report_axises=[1, 2, 3], C=num_classes)
kappameter = KappaMetrics()

for i, (model, state_dict) in enumerate(zip(models, state_dicts)):
    model.load_state_dict(state_dict['segmentator'])
    model.to(device)
    model.eval()
    print(f'model {i} has the best score: {state_dict["best_score"]:.3f}.')

with torch.no_grad():
    for i, ((img, gt), _, filename) in enumerate(dataloaders['val']):
        img, gt = img.to(device), gt.to(device)
        b = filename.__len__()
        preds = [model.predict(img, logit=False) for model in models]
        for j, pred in enumerate(preds):
            diceMeters[j].add(pred, gt)
            bdiceMeters[j].add(pred, gt)
        voting_preds = ensemble(preds)
        ensembleMeter.add(voting_preds, gt)
        bensembleMeter.add(voting_preds, gt)
        kappameter.add(predicts=[pred.max(1)[1] for pred in preds], target=voting_preds.max(1)[1],
                       considered_classes=[1, 2, 3])
## for 2D dice:
individual_result_dict = \
    {
        f'model_{i}': {f'DSC{j}': diceMeters[i].value()[1][0][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }
individual_std_dict = \
    {
        f'model_{i}': {f'DSC{j}': diceMeters[i].value()[1][1][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }

ensemble_result_dict = \
    {
        f'ensemble': {f'DSC{i}': ensembleMeter.value()[1][0][i].item() \
                      for i in range(num_classes)}
    }
ensemble_std_dict = \
    {
        f'ensemble': {f'DSC{i}': ensembleMeter.value()[1][1][i].item() for i in range(num_classes)}
    }

summary = pd.DataFrame({**ensemble_result_dict, **individual_result_dict})
summary_std = pd.DataFrame({**ensemble_std_dict, **individual_std_dict})
summary.to_csv(Path(args.input_dir) / 'summary.csv')
summary_std.to_csv(Path(args.input_dir) / 'summary_std.csv')

## for 3D dice:
individual_result_dict = \
    {
        f'model_{i}': {f'DSC{j}': bdiceMeters[i].value()[1][0][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }
individual_std_dict = \
    {
        f'model_{i}': {f'DSC{j}': bdiceMeters[i].value()[1][1][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }

ensemble_result_dict = \
    {
        f'ensemble': {f'DSC{i}': bensembleMeter.value()[1][0][i].item() \
                      for i in range(num_classes)}
    }
ensemble_std_dict = \
    {
        f'ensemble': {f'DSC{i}': bensembleMeter.value()[1][1][i].item() for i in range(num_classes)}
    }
summary = pd.DataFrame({**ensemble_result_dict, **individual_result_dict})
summary_std = pd.DataFrame({**ensemble_std_dict, **individual_std_dict})
summary.to_csv(Path(args.input_dir) / 'bsummary.csv')
summary_std.to_csv(Path(args.input_dir) / 'bsummary_std.csv')

diversity_dict = {f'Div{i}': kappameter.value()[i].item() for i in range(checkpoints.__len__())}
pd.Series(diversity_dict).to_csv(Path(args.input_dir) / 'div.csv')
