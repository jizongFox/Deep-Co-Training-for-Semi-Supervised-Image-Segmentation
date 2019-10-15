import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='input folder directory')
    parser.add_argument('--ensemble_method', default='soft', choices=('hard', 'soft'),
                        help='Ensemble method, either `soft` or `hard`')
    return parser.parse_args()


args = get_args()
input_dir = Path(args.input_dir)
assert input_dir.exists()
assert (input_dir / 'config.yml').exists()

import warnings
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from deepclustering.meters import HaussdorffDistance
from easydict import EasyDict
from torch import Tensor

from generalframework.dataset.ACDC_helper import get_ACDC_dataloaders
from generalframework.dataset.GM_helper import get_GMC_split_dataloders
from generalframework.dataset.spleen_helper import get_spleen_split_dataloders
from generalframework.metrics import KappaMetrics, DiceMeter
from generalframework.models import Segmentator
from generalframework.utils import probs2one_hot, class2one_hot, save_images, pred2class, dict_merge, tqdm_

warnings.filterwarnings('ignore')

with open(str(input_dir / 'config.yml'), 'r') as f:
    config = yaml.load(f.read())
    config = EasyDict(config)

checkpoints = list(input_dir.glob('best*.pth'))

if str(config.Dataset.root_dir).find('ACDC') >= 0:
    dataloaders = get_ACDC_dataloaders(config['Dataset'], config['Lab_Dataloader'], quite=True)
    dataloaders['val'].dataset.training = 'eval'
    report_axises = [1, 2, 3]
    patient_info = pd.read_csv('dataset/ACDC-all/patient_info.csv', header=None, index_col=[0])
elif str(config.Dataset.root_dir).find('GM') >= 0:
    config["Lab_Partitions"]["num_models"] = len(checkpoints)

    *_, val_dataloader = get_GMC_split_dataloders(config)
    dataloaders = {}
    dataloaders['val'] = val_dataloader
    dataloaders['val'].dataset.training = 'eval'
    report_axises = [0, 1]
    patient_info = None
elif str(config.Dataset.root_dir).lower().find("spleen") >= 0:
    *_, val_dataloader = get_spleen_split_dataloders(config)
    dataloaders = {}
    dataloaders['val'] = val_dataloader
    dataloaders['val'].dataset.training = 'eval'
    report_axises = [0, 1]
    patient_info = None

else:
    raise NotImplementedError


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


device = torch.device(config['Trainer']['device'])
ensemble = Ensembleway(args.ensemble_method)

models = load_models(checkpoints)

state_dicts = [torch.load(c, map_location=torch.device('cpu')) for c in checkpoints]

num_classes = state_dicts[0]['segmentator']['arch_dict']['num_classes']

diceMeters = [DiceMeter(method='2d', report_axises=report_axises, C=num_classes) for _ in range(checkpoints.__len__())]
bdiceMeters = [DiceMeter(method='3d', report_axises=report_axises, C=num_classes) for _ in range(checkpoints.__len__())]
hdMeter = [HaussdorffDistance(report_axises=report_axises, C=num_classes) for _ in range(checkpoints.__len__())]

ensembleMeter = DiceMeter(method='2d', report_axises=report_axises, C=num_classes)
bensembleMeter = DiceMeter(method='3d', report_axises=report_axises, C=num_classes)
hdensembleMeter = HaussdorffDistance(report_axises=report_axises, C=num_classes)
kappameter = KappaMetrics()

for i, (model, state_dict) in enumerate(zip(models, state_dicts)):
    model.load_state_dict(state_dict['segmentator'])
    model.to(device)
    model.eval()
    print(f'model {i} has the best score: {state_dict["best_score"]:.3f}.')

with torch.no_grad():
    for i, ((img, gt), _, path) in enumerate(tqdm_(dataloaders['val'])):
        img, gt = img.to(device), gt.to(device)
        patient_id = list(set([p.split('_')[0] for p in path]))[0]
        preds = [model.predict(img, logit=False) for model in models]
        for j, pred in enumerate(preds):
            diceMeters[j].add(pred, gt)
            bdiceMeters[j].add(pred, gt)
            hdMeter[j].add(class2one_hot(pred2class(pred), C=num_classes), class2one_hot(gt.squeeze(1), C=num_classes),
                           voxelspacing=patient_info.loc[patient_id].to_numpy()[
                               0] if patient_info is not None else None)

            save_images(pred2class(pred), names=path, root=args.input_dir, mode='val', iter=1000,
                        seg_num=str(j))

        voting_preds = ensemble(preds)

        save_images(pred2class(voting_preds), names=path, root=args.input_dir, mode='val', iter=1000,
                    seg_num='voting')

        ensembleMeter.add(voting_preds, gt)
        bensembleMeter.add(voting_preds, gt)
        hdensembleMeter.add(class2one_hot(pred2class(voting_preds), C=num_classes),
                            class2one_hot(gt.squeeze(1), C=num_classes))
        kappameter.add(predicts=[pred.max(1)[1] for pred in preds], target=voting_preds.max(1)[1],
                       considered_classes=report_axises)
        # todo: add kappa2 score

# for 2D dice:
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

# for 3D dice:
individual_result_dict = \
    {
        f'model_{i}': {f'DSC{j}': bdiceMeters[i].value()[1][0][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }
individual_HD_dict = \
    {
        f'model_{i}': {f'HD{j}': hdMeter[i].value()[1][0][j].item() \
                       for j in range(num_classes)} for i in range(models.__len__())
    }
individual_result_dict = dict_merge(individual_result_dict, individual_HD_dict, re=True)

individual_std_dict = \
    {
        f'model_{i}': {f'DSC{j}': bdiceMeters[i].value()[1][1][j].item() \
                       for j in range(num_classes)
                       } for i in range(models.__len__())
    }

ensemble_result_dict = \
    {
        f'ensemble': {f'DSC{i}': bensembleMeter.value()[1][0][i].item() \
                      for i in range(num_classes)}
    }

ensemble_HD_dict = \
    {
        f'ensemble': {f'HD{i}': hdensembleMeter.value()[1][0][i].item() \
                      for i in range(num_classes)}
    }
ensemble_result_dict = dict_merge(ensemble_result_dict, ensemble_HD_dict, re=True)
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

# record val_logs

pd.DataFrame(bensembleMeter.log.cpu().numpy()).to_csv(Path(args.input_dir, 'log_3Ddice_ensemble.csv'))
pd.DataFrame(ensembleMeter.log.cpu().numpy()).to_csv(Path(args.input_dir, 'log_2Ddice_ensemble.csv'))
for s in range(len(checkpoints)):
    pd.DataFrame(diceMeters[s].log.cpu().numpy()).to_csv(Path(args.input_dir, f'log_2Ddice_model_{s}.csv'))
    pd.DataFrame(bdiceMeters[s].log.cpu().numpy()).to_csv(Path(args.input_dir, f'log_3Ddice_model_{s}.csv'))
