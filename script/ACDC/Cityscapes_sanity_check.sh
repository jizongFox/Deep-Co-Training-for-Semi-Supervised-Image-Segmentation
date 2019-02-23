#!/usr/bin/env bash
set -e
cd ..
max_epoch=300
net=enet
num_model=2
ratio=0.95
logdir=cityscapes/$net"_cotraining_"$num_model"_models_"$ratio

FS(){
gpu=$1
currentfoldername=FS
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_cityscapes_cotraining.py Lab_Partitions.split_ratio=1 Lab_Partitions.num=$num_model StartTraining.train_jsd=False StartTraining.train_adv=False \
Trainer.max_epoch=$max_epoch Trainer.save_dir=runs/$logdir/$currentfoldername
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

PS(){
gpu=$1
ratio=$2
currentfoldername="PS"
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_cityscapes_cotraining.py Lab_Partitions.split_ratio=$ratio Lab_Partitions.num=$num_model StartTraining.train_jsd=False StartTraining.train_adv=False \
Trainer.max_epoch=$max_epoch Trainer.save_dir=runs/$logdir/$currentfoldername
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD(){
gpu=$1
ratio=$2
currentfoldername="JSD"
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_cityscapes_cotraining.py Lab_Partitions.split_ratio=$ratio Lab_Partitions.num=$num_model StartTraining.train_jsd=True StartTraining.train_adv=False \
Trainer.max_epoch=$max_epoch Trainer.save_dir=runs/$logdir/$currentfoldername
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

ADV(){
gpu=$1
ratio=$2
currentfoldername="ADV"
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_cityscapes_cotraining.py Lab_Partitions.split_ratio=$ratio Lab_Partitions.num=$num_model StartTraining.train_jsd=True StartTraining.train_adv=False \
Trainer.max_epoch=$max_epoch Trainer.save_dir=runs/$logdir/$currentfoldername
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD_ADV(){
gpu=$1
ratio=$2
currentfoldername="JSD_ADV"
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_cityscapes_cotraining.py Lab_Partitions.split_ratio=$ratio Lab_Partitions.num=$num_model StartTraining.train_jsd=True StartTraining.train_adv=True \
Trainer.max_epoch=$max_epoch Trainer.save_dir=runs/$logdir/$currentfoldername
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

Summary(){
subfolder=$1
gpu=$2
echo CUDA_VISIBLE_DEVICES=$gpu python Summary_city.py --input_dir archives/$logdir/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary_city.py --input_dir archives/$logdir/$subfolder
}

mkdir -p archives/$logdir
#
FS 0 $ratio
PS 0 $ratio
JSD 0 $ratio
ADV 0 $ratio
JSD_ADV 0 $ratio

python generalframework/postprocessing/plot_cityscapes.py --folders archives/$logdir/FS archives/$logdir/PS archives/$logdir/FS archives/$logdir/JSD archives/$logdir/ADV archives/$logdir/JSD_ADV \
--file=val_class_IoU.npy --postfix=model0 --num_seg=0

python generalframework/postprocessing/plot_cityscapes.py --folders archives/$logdir/FS archives/$logdir/PS archives/$logdir/FS archives/$logdir/JSD archives/$logdir/ADV archives/$logdir/JSD_ADV \
--file=val_class_IoU.npy --postfix=model1 --num_seg=1

Summary FS 0
Summary PS 0
Summary JSD 0
Summary ADV 0
Summary JSD_ADV 0