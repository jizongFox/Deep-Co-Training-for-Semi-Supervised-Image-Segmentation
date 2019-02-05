from random import random
from typing import Dict

import pandas as pd
import yaml
from tensorboardX import SummaryWriter
from operator import itemgetter

from generalframework import ModelMode
from ..metrics.iou import IoU
from .trainer import Trainer
from ..loss import CrossEntropyLoss2d, KL_Divergence_2D
from ..models import Segmentator
from ..scheduler import RampScheduler
from ..utils.AEGenerator import *
from ..utils.metrics import scores
from ..utils.utils import *


class CoTrainer_City(Trainer):

    def __init__(self, segmentators: List[Segmentator], labeled_dataloaders: List[DataLoader],
                 unlabeled_dataloader: DataLoader, val_dataloader: DataLoader, criterions: Dict[str, nn.Module],
                 max_epoch: int = 100, save_dir: str = 'tmp', device: str = 'cpu',
                 axises: List[int] = None, checkpoint: str = None, metricname: str = 'metrics.csv',
                 lambda_cot_max: int = 10, lambda_adv_max: float = 0.5, ramp_up_mult: float = -5,
                 epoch_max_ramp: int = 80, whole_config=None) -> None:

        self.max_epoch = max_epoch
        self.segmentators = segmentators
        self.labeled_dataloaders = labeled_dataloaders
        self.unlabeled_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader

        # N segmentators should be consist with N+1 dataloders
        # (N for labeled data and N+2 th for unlabeled dataset)
        assert self.segmentators.__len__() == self.labeled_dataloaders.__len__()
        assert self.segmentators.__len__() >= 1
        # the sgementators and dataloaders must be different instance
        assert set(map_(id, self.segmentators)).__len__() == self.segmentators.__len__()
        assert set(map_(id, self.labeled_dataloaders)).__len__() == self.segmentators.__len__()

        # labeled_dataloaders should have the same number of images
        # assert set(map_(lambda x: len(x.dataset), self.labeled_dataloaders)).__len__() == 1
        # assert set(map_(lambda x: len(x), self.labeled_dataloaders)).__len__() == 1

        self.criterions = criterions
        assert set(self.criterions.keys()) == set(['jsd', 'sup', 'adv'])

        self.save_dir = Path(save_dir)
        # assert not (self.save_dir.exists() and checkpoint is None), f'>> save_dir: {self.save_dir} exits.'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(save_dir)
        # save the whole new config to the save_dir
        if whole_config:
            with open(Path(self.save_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(whole_config, outfile, default_flow_style=True)

        self.device = torch.device(device)
        self.C = self.segmentators[0].arch_params['num_classes']
        if axises is None:
            axises = list(range(self.C))
        self.axises = axises
        self.best_scores = np.zeros(self.segmentators.__len__())
        self.start_epoch = 0
        self.metricname = metricname

        # scheduler
        self.cot_scheduler = RampScheduler(max_epoch=epoch_max_ramp, max_value=lambda_cot_max, ramp_mult=ramp_up_mult)
        self.adv_scheduler = RampScheduler(max_epoch=epoch_max_ramp, max_value=lambda_adv_max, ramp_mult=ramp_up_mult)

        if checkpoint is not None:
            # todo
            self._load_checkpoint(checkpoint)

        self.to(self.device)

    def to(self, device: torch.device):
        [segmentator.to(device) for segmentator in self.segmentators]
        [criterion.to(device) for _, criterion in self.criterions.items()]

    def start_training(self, train_jsd=False, train_adv=False, save_train=False, save_val=False):
        # prepare for something:
        S = len(self.segmentators)
        train_b = max(map_(len, self.labeled_dataloaders))
        val_b: int = len(self.val_dataloader)

        metrics = {"val_loss": torch.zeros((self.max_epoch, val_b, S), device=self.device).type(torch.float32),
                   "val_mean_IoU": torch.zeros((self.max_epoch, S), device=self.device).type(torch.float32),
                   "val_mean_Acc": torch.zeros((self.max_epoch, S), device=self.device).type(torch.float32),
                   "val_class_IoU": torch.zeros((self.max_epoch, S, self.C), device=self.device).type(
                       torch.float32),

                   "train_loss": torch.zeros((self.max_epoch, train_b, S), device=self.device).type(torch.float32),
                   "train_jsd_loss": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_adv_loss": torch.zeros((self.max_epoch, train_b), device=self.device).type(torch.float32),
                   "train_mean_IoU": torch.zeros((self.max_epoch, S), device=self.device).type(torch.float32),
                   "train_mean_Acc": torch.zeros((self.max_epoch, S), device=self.device).type(torch.float32),
                   "train_class_IoU": torch.zeros((self.max_epoch, S, self.C), device=self.device).type(
                       torch.float32)
                   }

        for epoch in range(self.start_epoch, self.max_epoch):

            train_mean_Acc, _, \
            train_mean_IoU, \
            train_class_IoU, \
            train_loss, \
            train_jsd_loss, \
            train_adv_loss = self._train_loop(labeled_dataloaders=self.labeled_dataloaders,
                                              unlabeled_dataloader=self.unlabeled_dataloader,
                                              epoch=epoch,
                                              mode=ModelMode.TRAIN,
                                              save=save_train,
                                              train_jsd=train_jsd,
                                              train_adv=train_adv
                                              )

            with torch.no_grad():
                val_mean_Acc, _, \
                val_mean_IoU, \
                val_class_IoU, \
                val_loss = self._eval_loop(val_dataloader=self.val_dataloader,
                                           epoch=epoch,
                                           mode=ModelMode.EVAL,
                                           save=save_val)

            self.schedulerStep()

            for k, v in metrics.items():
                try:
                    v[epoch] = eval(k) if torch.is_tensor(eval(k)) else torch.Tensor(eval(k))
                except:
                    v[epoch] = torch.stack(eval(k))

            for k, v in metrics.items():
                np.save(self.save_dir / f'{k}.npy', v.data.cpu().numpy())

            # writer = pd.ExcelWriter(Path(self.save_dir, self.metricname.replace('csv', 'xlsx')), engine='openpyxl')
            # for s in range(self.segmentators.__len__()):
            #     df = pd.DataFrame(
            #         {
            #             **{f"train_dice_{i}": metrics["train_dice"].mean(1)[:, s, i] for i in self.axises},
            #             **{f"train_unlab_dice_{i}": metrics["train_unlab_dice"].mean(1)[:, s, i] for i in
            #                self.axises},
            #             **{f"val_dice_{i}": metrics["val_dice"].mean(1)[:, s, i] for i in self.axises},
            #             **{f"val_batch_dice_{i}": metrics["val_batch_dice"].mean(1)[:, s, i] for i in self.axises}
            #         })
            #     # the saved metrics are with only axis==3, as the foreground dice.
            #
            #     df.to_csv(Path(self.save_dir, self.metricname.replace('.csv', f'_{s}.csv')), float_format="%.4f",
            #               index_label="epoch")
            #     df.to_excel(excel_writer=writer, sheet_name=f'Seg_{s}', encoding="utf-8", index_label='epoch',
            #                 float_format="%.4f")
            # writer.save()
            # writer.close()
            current_metric = val_dice[:, :, self.axises].mean([0, 2])
            self.checkpoint(current_metric, epoch)

    def _train_loop(self, labeled_dataloaders: List[DataLoader], unlabeled_dataloader: DataLoader, epoch: int,
                    mode: ModelMode, save: bool, augment_labeled_data=False, augment_unlabeled_data=False,
                    train_jsd=False, train_adv=False):
        [segmentator.set_mode(mode) for segmentator in self.segmentators]
        for l_dataloader in labeled_dataloaders:
            l_dataloader.dataset.set_mode(ModelMode.TRAIN)
        unlabeled_dataloader.dataset.training = ModelMode.TRAIN if augment_unlabeled_data else ModelMode.EVAL

        assert self.segmentators[0].training == True
        assert self.labeled_dataloaders[0].dataset.training == ModelMode.TRAIN \
            if augment_labeled_data else ModelMode.EVAL
        assert self.unlabeled_dataloader.dataset.training == ModelMode.TRAIN \
            if augment_unlabeled_data else ModelMode.EVAL

        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        # Here the concept of epoch is defined as the epoch
        n_batch = max(map_(len, self.labeled_dataloaders))
        S = len(self.segmentators)
        metrics = [IoU(self.C, ignore_index=255) for _ in range(S)]
        sup_loss_log = torch.zeros(n_batch, S)
        jsd_loss_log = torch.zeros(n_batch)
        adv_loss_log = torch.zeros(n_batch)

        # build fake_iterator
        fake_labeled_iterators = [iterator_(dcopy(x)) for x in labeled_dataloaders]
        fake_labeled_iterators_adv = [iterator_(dcopy(x)) for x in labeled_dataloaders]

        fake_unlabeled_iterator = iterator_(dcopy(unlabeled_dataloader))
        fake_unlabeled_iterator_adv = iterator_(dcopy(unlabeled_dataloader))

        n_batch_iter = tqdm_(range(n_batch))

        for batch_num in n_batch_iter:
            # for labeled data update
            for enu_lab in range(len(fake_labeled_iterators)):
                [[img, gt], _, path] = fake_labeled_iterators[enu_lab].__next__()
                img, gt = img.to(self.device), gt.to(self.device)
                # backward and update when the mode = ModelMode.TRAIN
                pred, sup_loss = self.segmentators[enu_lab].update(img, gt, criterion=self.criterions.get('sup'),
                                                                   mode=ModelMode.TRAIN)

                metrics[enu_lab].add(predicted=pred, target=gt)

                sup_loss_log[batch_num] = sup_loss.detach()

                if save:
                    save_images(pred2class(pred),
                                names=map_(lambda x: Path(x).name, path),
                                root=self.save_dir,
                                mode='train',
                                iter=epoch,
                                seg_num=str(enu_lab))

            if train_jsd:
                # for unlabeled data update
                [[unlab_img, unlab_gt], _, path] = fake_unlabeled_iterator.__next__()
                unlab_img, unlab_gt = unlab_img.to(self.device), unlab_gt.to(self.device)
                unlab_preds: List[Tensor] = map_(lambda x: x.predict(unlab_img, logit=False), self.segmentators)
                assert unlab_preds.__len__() == self.segmentators.__len__()

                # function for JSD
                jsdloss_2D = self.criterions.get('jsd')(unlab_preds)
                assert jsdloss_2D.shape[0] == unlab_img.shape[0]
                assert jsdloss_2D.shape[1] == unlab_img.shape[2]
                assert jsdloss_2D.shape[2] == unlab_img.shape[3]
                jsdloss = jsdloss_2D.mean()
                jsd_loss_log[batch_num] = jsdloss.detach()

                if save:
                    [save_images(probs2class(prob),
                                 names=map_(lambda x: Path(x).name, path),
                                 root=self.save_dir,
                                 mode='unlab',
                                 iter=epoch,
                                 seg_num=str(i))
                     for i, prob in enumerate(unlab_preds)]

                # zero grad
                map_(lambda x: x.optimizer.zero_grad(), self.segmentators)
                loss = jsdloss * self.cot_scheduler.value
                loss.backward()
                map_(lambda x: x.optimizer.step(), self.segmentators)

            # adversarial loss:
            if train_adv:
                assert self.segmentators.__len__() == 2, 'only implemented for 2 segmentators'
                ### TODO for more than 2 segmentators
                adv_losses = []

                ## draw first term from labeled1 or unlabeled
                img, img_adv = None, None
                if random() > 0.5:
                    [[img, gt], _, _] = fake_labeled_iterators_adv[0].__next__()
                    img, gt = img.to(self.device), gt.to(self.device)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        img_adv, _ = FSGMGenerator(self.segmentators[0].torchnet, eplision=0.05) \
                            (dcopy(img), gt, criterion=self.criterion['sup'])
                else:
                    [[img, _], _, _] = fake_unlabeled_iterator_adv.__next__()
                    img = img.to(self.device)
                    img_adv, _ = VATGenerator(self.segmentators[0].torchnet, eplision=0.05)(dcopy(img))
                assert img.shape == img_adv.shape
                adv_pred = self.segmentators[1].predict(img_adv, logit=False)
                real_pred = self.segmentators[0].predict(img, logit=False)
                adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred))

                if random() > 0.5:
                    [[img, gt], _, _] = fake_labeled_iterators_adv[1].__next__()
                    img, gt = img.to(self.device), gt.to(self.device)
                    img_adv, _ = FSGMGenerator(self.segmentators[1].torchnet, eplision=0.05) \
                        (img, gt, criterion=CrossEntropyLoss2d())
                else:
                    [[img, _], _, _] = fake_unlabeled_iterator_adv.__next__()
                    img = img.to(self.device)
                    img_adv, _ = VATGenerator(self.segmentators[1].torchnet, eplision=0.05)(img)

                adv_pred = self.segmentators[0].predict(img_adv, logit=False)
                real_pred = self.segmentators[1].predict(img, logit=False)
                adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred.detach()))

                adv_loss = self._adv_training(segmentators=itemgetter(*choice)(self.segmentators),
                                              lab_data_iterators=itemgetter(*choice)(fake_labeled_iterators_adv),
                                              unlab_data_iterator=fake_unlabeled_iterator_adv,
                                              eplision=0.005)
                map_(lambda x: x.optimizer.zero_grad(), self.segmentators)
                adv_loss *= self.adv_scheduler.value
                adv_loss.backward()
                map_(lambda x: x.optimizer.step(), self.segmentators)
                adv_loss_log[batch_num] = adv_loss.detach()

            mean_iou_dict = {f'{i}_mIoU': metrics[i].value()['Mean_IoU'] for i in range(S)}

            stat_dict = {
                **mean_iou_dict,
                **{f'{i}_supls': sup_loss_log[:batch_num + 1, i].mean().item() for i in range(S)},
                **{'jsdls': jsd_loss_log[:batch_num + 1].mean().item()},
                **{'advls': adv_loss_log[:batch_num + 1].mean().item()}
            }
            # to delete null dicts
            nice_dict = {k: f"{v:.2f}" for (k, v) in stat_dict.items() if v != 0 or v != float(np.nan)}
            #
            n_batch_iter.set_description(
                f'{"tls" if mode == ModelMode.TRAIN else "vlos"}')
            n_batch_iter.set_postfix(nice_dict)  # using average value of the dict
        print(f"{desc} " + ', '.join(f"{k}:{v}" for (k, v) in nice_dict.items()))

        return map_(lambda x: x.value()["Mean_Acc"], metrics), map_(lambda x: x.value()["FreqW_Acc"], metrics), map_(
            lambda x: x.value()["Mean_IoU"], metrics), map_(lambda x: x.value()["Class_IoU"],
                                                            metrics), sup_loss_log, jsd_loss_log, adv_loss_log

        #     #
        # self.upload_dicts('labeled dataset', lab_dsc_dict, epoch)
        # self.upload_dicts('unlabeled dataset', unlab_dsc_dict, epoch)

        ## make sure that the nice dict is for labeled dataset
        # nice_dict = dict_merge(lab_dsc_dict, lab_mean_dict, re=True)
        # print(
        #     f"{desc} " + ', '.join([f'{k}_{k_}: {v[k_]:.2f}' for k, v in nice_dict.items() for k_ in v.keys()])
        # )
        # return coef_dice, unlabel_coef_dice

    def _eval_loop(self, val_dataloader: DataLoader,
                   epoch: int,
                   mode: ModelMode = ModelMode.EVAL,
                   save: bool = False
                   ):
        [segmentator.set_mode(mode) for segmentator in self.segmentators]
        val_dataloader.dataset.set_mode(ModelMode.EVAL)
        assert self.segmentators[0].training == False
        desc = f">>   Training   ({epoch})" if mode == ModelMode.TRAIN else f">> Validating   ({epoch})"
        S = self.segmentators.__len__()
        n_batch = len(val_dataloader)
        loss_log = torch.zeros(n_batch, S)
        val_dataloader = tqdm_(val_dataloader)
        metrics = [IoU(self.C, ignore_index=255) for _ in range(S)]

        dsc_dict = {}
        nice_dict = {}

        for batch_num, [(img, gt), _, path] in enumerate(val_dataloader):
            img, gt = img.to(self.device), gt.to(self.device)
            preds = map_(lambda x: x.predict(img, logit=True), self.segmentators)
            sup_loss = map_(lambda x: self.criterions.get('sup')(x, gt.squeeze(1)), preds)
            for i in range(len(preds)):
                metrics[i].add(predicted=preds[i], target=gt)
                loss_log[batch_num][i] = sup_loss[i].detach()

            if save:
                [save_images(pred2class(pred), names=path, root=self.save_dir, mode='eval', seg_num=str(i), iter=epoch)
                 for i, pred in enumerate(preds)]

            mean_iou_dict = {f'{i}_mIoU': metrics[i].value()['Mean_IoU'] for i in range(S)}

            stat_dict = {**mean_iou_dict}
            # to delete null dicts
            nice_dict = {k: f"{v:.2f}" for (k, v) in stat_dict.items() if v != 0 or v != float(np.nan)}
            val_dataloader.set_description(
                f'{"tls" if mode == ModelMode.TRAIN else "vlos"}')
            val_dataloader.set_postfix(nice_dict)  # using average value of the dict

        # self.upload_dicts('val_data', dsc_dict, epoch)

        print(f"{desc} " + ', '.join(f"{k}:{v}" for (k, v) in nice_dict.items()))

        return map_(lambda x: x.value()["Mean_Acc"], metrics), map_(lambda x: x.value()["FreqW_Acc"], metrics), map_(
            lambda x: x.value()["Mean_IoU"], metrics), map_(lambda x: x.value()["Class_IoU"],
                                                            metrics), loss_log

    def _adv_training(self, segmentators: List[Segmentator],
                      lab_data_iterators: List[iterator_], unlab_data_iterator: iterator_,
                      eplision: float = 0.05):
        assert segmentators.__len__() == 2, 'only implemented for 2 segmentators'
        adv_losses = []
        ## draw first term from labeled1 or unlabeled
        img, img_adv = None, None
        if random() > 0.5:
            [[img, gt], _, _] = lab_data_iterators[0].__next__()
            img, gt = img.to(self.device), gt.to(self.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img_adv, _ = FSGMGenerator(self.segmentators[0].torchnet, eplision=eplision) \
                    (dcopy(img), gt, criterion=self.criterions['sup'])
        else:
            [[img, _], _, _] = unlab_data_iterator.__next__()
            img = img.to(self.device)
            img_adv, _ = VATGenerator(self.segmentators[0].torchnet, eplision=eplision)(dcopy(img))
        assert img.shape == img_adv.shape
        adv_pred = segmentators[1].predict(img_adv, logit=False)
        real_pred = segmentators[0].predict(img, logit=False)
        adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred))

        if random() > 0.5:
            [[img, gt], _, _] = lab_data_iterators[1].__next__()
            img, gt = img.to(self.device), gt.to(self.device)
            img_adv, _ = FSGMGenerator(self.segmentators[1].torchnet, eplision=eplision) \
                (img, gt, criterion=CrossEntropyLoss2d())
        else:
            [[img, _], _, _] = unlab_data_iterator.__next__()
            img = img.to(self.device)
            img_adv, _ = VATGenerator(self.segmentators[1].torchnet, eplision=eplision)(img)

        adv_pred = segmentators[0].predict(img_adv, logit=False)
        real_pred = segmentators[1].predict(img, logit=False)
        adv_losses.append(KL_Divergence_2D(reduce=True)(adv_pred, real_pred.detach()))

        adv_loss = sum(adv_losses) / adv_losses.__len__()

        return adv_loss

    def upload_dicts(self, name, dicts, epoch):
        for k, v in dicts.items():
            name_ = name + '/' + k
            self.upload_dict(name_, v, epoch)

    def upload_dict(self, name, dict, epoch):
        self.writer.add_scalars(name, dict, epoch)

    def schedulerStep(self):
        for segmentator in self.segmentators:
            segmentator.schedulerStep()
        self.cot_scheduler.step()
        self.adv_scheduler.step()

    def checkpoint(self, metric, epoch, filename='best.pth'):
        assert isinstance(metric, Tensor)
        assert metric.__len__() == self.segmentators.__len__()
        for i, score in enumerate(metric):
            # slack variable:
            self.best_score = self.best_scores[i]
            self.segmentator = self.segmentators[i]
            super().checkpoint(score, epoch, filename=f'best_{i}.pth')
            self.best_scores[i] = self.best_score
