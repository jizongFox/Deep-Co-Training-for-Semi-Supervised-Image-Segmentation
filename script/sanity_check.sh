#!/usr/bin/env bash
set -e
cd ..
max_peoch=120
data_aug=None
net=enet
logdir=cardiac/$net"_FS_sanity_check_2model"
mkdir -p archives/$logdir
## Fulldataset baseline

FS(){
gpu=$1
currentfoldername=FS
rm -rf runs/$logdir/$currentfoldername
echo cdCUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=False StartTraining.train_jsd=False Model_num=2 Lab_Partitions.label="[[1,101],[1,101]]"
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

Partial(){
gpu=$1
currentfoldername=PS
rm -rf runs/$logdir/$currentfoldername
echo CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=240 \
Dataset.augment=$data_aug StartTraining.train_adv=False StartTraining.train_jsd=False \
Model_num=2 Lab_Partitions.label="[[1,41],[21,61]]" Lab_Partitions.unlabel="[61,101]"
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}
FS 1 & Partial 2

#jsd(){
#gpu=$1
#currentfoldername=PS
#rm -rf runs/$logdir/$currentfoldername
#CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
#Dataset.augment=$data_aug StartTraining.train_adv=False StartTraining.train_jsd=True \
#Model_num=3 Lab_Partitions.label="[[1,41],[21,61]]" Lab_Partitions.unlabel="[61,101]"
#rm -rf archives/$logdir/$currentfoldername
#mv -f runs/$logdir/$currentfoldername archives/$logdir
#}
#
#adv(){
#gpu=$1
#currentfoldername=PS
#rm -rf runs/$logdir/$currentfoldername
#CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
#Dataset.augment=$data_aug StartTraining.train_adv=True StartTraining.train_jsd=False \
#Model_num=3 Lab_Partitions.label="[[1,41],[21,61]]" Lab_Partitions.unlabel="[61,101]"
#rm -rf archives/$logdir/$currentfoldername
#mv -f runs/$logdir/$currentfoldername archives/$logdir
#}