#!/usr/bin/env bash
set -e
max_peoch=200
data_aug=None
logdir=cardiac/unet_FS_sanity_check
mkdir -p archives/$logdir
## Fulldataset baseline

fs(){
currentfoldername=train
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

vat(){
### Partial dataset baseline
currentfoldername=vat
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=2 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

co_train(){
currentfoldername=cotrain
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=3 python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=False StartTraining.train_jsd=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

fs & vat & co_train
