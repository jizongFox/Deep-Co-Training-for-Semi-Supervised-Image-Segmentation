#!/usr/bin/env bash
set -e
max_peoch=100
data_aug=None
logdir=cardiac/unet_VAT
mkdir -p archives/$logdir
## Fulldataset baseline

fs(){
currentfoldername=FS_fulldata
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=0 python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

partial(){
### Partial dataset baseline
currentfoldername=FS_partial
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

adv(){
currentfoldername=adv
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=2 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}


fs & partial & adv

python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_dice.npy --axis 1 2 3 --postfix=test
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_batch_dice.npy --axis 1 2 3 --postfix=test