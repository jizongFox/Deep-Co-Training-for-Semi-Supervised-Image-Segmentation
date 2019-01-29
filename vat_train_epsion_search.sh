#!/usr/bin/env bash
set -e
max_peoch=100
data_aug=None
net=enet
logdir=cardiac/$net"_VAT"_epsilions
mkdir -p archives/$logdir


adv(){
gpuid=$1
eps=$2
echo gpuid $gpuid
echo eps $eps
currentfoldername=adv_eps_$eps
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpuid python train_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug   Arch.name=$net  StartTraining.train_adv=True StartTraining.adv_config.eplision=$eps
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

adv 1 0.0 &
adv 2 0.001 &
adv 1 0.005 &
adv 2 0.01

adv 1 0.05 &
adv 2 0.1 &
adv 1 0.5

#python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_dice.npy --axis 1 2 3 --postfix=test
#python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/adv/  --file val_batch_dice.npy --axis 1 2 3 --postfix=test