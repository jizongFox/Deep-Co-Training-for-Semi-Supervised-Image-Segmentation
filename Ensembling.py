import torch
import torch.nn.functional as F
from pprint import pprint
import yaml
from generalframework.dataset import get_ACDC_dataloaders
from generalframework.utils import yaml_parser, dict_merge, dice_coef, probs2one_hot, class2one_hot, map_
from generalframework.models import Segmentator
from typing import List
from torch import Tensor
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser_args = yaml_parser()
print('->>Input args:')
pprint(parser_args)
with open('config/ensembling_config.yaml', 'r') as f:
    config = yaml.load(f.read())
print('->> Merged Config:')
config = dict_merge(config, parser_args, True)
# pprint(config)

dataloaders = get_ACDC_dataloaders(config['Dataset'], config['Dataloader'], quite=True)
dataloaders['val'].dataset.training = 'eval'


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

models = [Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler']) for _ in
          range(config['Checkpoints'].__len__())]

state_dicts = [torch.load(c, map_location=torch.device('cpu')) for c in config['Checkpoints']]

val_b: int = len(dataloaders['val'])
val_n: int = val_b * dataloaders['val'].batch_size if dataloaders['val'].drop_last \
    else len(dataloaders['val'].dataset)
coef_dice = torch.zeros(val_n, len(models) + 1, config['Arch']['num_classes'])

for i, (model, state_dict) in enumerate(zip(models, state_dicts)):
    model.load_state_dict(state_dict['segmentator'])
    model.to(device)
    model.eval()
    print(f'model {i} has the best score: {state_dict["best_score"]:.3f}.')


with torch.no_grad():
    done = 0
    for i, ((img, gt), _, filename) in tqdm(enumerate(dataloaders['val'])):
        img, gt = img.to(device), gt.to(device)
        b = filename.__len__()
        preds = [model.predict(img, logit=False) for model in models]
        for j, pred in enumerate(preds):
            coef_dice[done:done + b, j] = dice_coef(*toOneHot(pred, gt))

        preds_ = ensemble(preds)
        coef_dice[done:done + b, -1] = dice_coef(*toOneHot(preds_, gt))
        done += b

    individual_result_dict = {
        f'model_{i}': {f'DSC{j}': coef_dice[:, i, j].mean().item() for j in range(config['Arch']['num_classes'])} for i
        in
        range(models.__len__())}
    ensemble_result_dict = {
        f'ensemble': {f'DSC{i}': coef_dice[:, -1, i].mean().item() for i in range(config['Arch']['num_classes'])}}
    print(f'evaluation done with:\n')
    pprint(individual_result_dict)
    pprint(ensemble_result_dict)

