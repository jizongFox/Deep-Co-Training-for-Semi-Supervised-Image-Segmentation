#!/usr/bin/env bash
max_peoch=100
data_aug=None
logdir=runs
mkdir -p archives
## Fulldataset baseline
rm -rf $logdir/FS_fulldataset
python train.py Trainer.save_dir=$logdir/FS_fulldataset Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug
rm -rf archives/FS_fulldataset
mv -f $logdir/FS_fulldataset archives/

## Partial dataset baseline
rm -rf $logdir/FS_partialdataset
python train_cotraining.py Trainer.save_dir=$logdir/FS_partialdataset Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=False, StartTraining.train_adv=False \

rm -rf archives/FS_partialdataset
mv -f $logdir/FS_partialdataset archives/
