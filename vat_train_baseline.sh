#!/usr/bin/env bash
set -e
max_peoch=100
data_aug=None
logdir=cardiac/unet_VAT
mkdir -p archives/$logdir
## Fulldataset baseline

currentfoldername=FS_fulldata
rm -rf $logdir/$currentfoldername
python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir

## Partial dataset baseline
currentfoldername=FS_partial
rm -rf $logdir/$currentfoldername
python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir


currentfoldername=adv
rm -rf $logdir/$currentfoldername
python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True \
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir


python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_dice.npy --axis 1 2 3 --postfix=test
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_batch_dice.npy --axis 1 2 3 --postfix=test