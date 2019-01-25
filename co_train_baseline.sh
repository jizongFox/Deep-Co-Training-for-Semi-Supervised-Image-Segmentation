#!/usr/bin/env bash
set -e
max_peoch=100
data_aug=None
logdir=cardiac/unet_co_training
mkdir -p archives/$logdir
fs(){
## Fulldataset baseline
currentfoldername=FS_fulldata
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

partial(){
## Partial dataset baseline
currentfoldername=FS_partial
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=False StartTraining.train_adv=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

onlyjsd(){
currentfoldername=only_jsd
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=2 python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=True StartTraining.train_adv=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

only_adv(){
currentfoldername=only_adv
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=2 python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=False StartTraining.train_adv=True
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

jsd_adv(){
currentfoldername=jsd_adv
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=3 python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=True StartTraining.train_adv=True
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

gpu1(){
fs
partial
}
gpu2(){
onlyjsd
only_adv
}
gpu3(){
jsd_adv
}
gpu1 & gpu2 & gpu3

python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/only_jsd/ archives/$logdir/only_adv/ archives/$logdir/jsd_adv/ --file val_dice.npy --axis 1 2 3 --postfix=test
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/only_jsd/ archives/$logdir/only_adv/ archives/$logdir/jsd_adv/ --file val_batch_dice.npy --axis 1 2 3 --postfix=test