#!/usr/bin/env bash
set -e

cd ..
max_peoch=100
data_aug=None
net=unet
logdir=cardiac/$net"_VAT"
mkdir -p archives/$logdir
## Fulldataset baseline

pretrain()
{
### Partial dataset baseline
currentfoldername=Partail_pretrain
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=15 \
Dataset.augment=$data_aug  StartTraining.train_adv=False  Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

fs(){
currentfoldername=FS_fulldata
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=1 python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

partial(){
### Partial dataset baseline
currentfoldername=FS_partial
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=2 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False  Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

adv(){
currentfoldername=adv
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=3 python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True  Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}


adv & fs & adv & pretrain & partial

python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_dice.npy --axis 1 2 3 --postfix=test
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_batch_dice.npy --axis 1 2 3 --postfix=test