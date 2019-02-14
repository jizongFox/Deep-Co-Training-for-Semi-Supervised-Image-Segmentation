#!/usr/bin/env bash
set -e
cd ..
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

max_peoch=200
data_aug=None
net=enet
logdir=cardiac/$net"_2modelbaseline_samepartitions"


FS(){
gpu=$1
currentfoldername=FS
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,101],[1,101]]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

Partial(){
gpu=$1
currentfoldername=PS
rm -rf runs/$logdir/$currentfoldername

CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}



Partial_alldata(){
gpu=$1
currentfoldername=PS_alldata
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD(){
gpu=$1
currentfoldername=JSD
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=True \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

ADV(){
gpu=$1
currentfoldername=ADV
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD_ADV(){
gpu=$1
currentfoldername=JSD_ADV
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

Summary(){
subfolder=$1
gpu=$2
echo CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir archives/$logdir/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir archives/$logdir/$subfolder
}

mkdir -p archives/$logdir

FS 2
Partial 2
#Partial_alldata 2
JSD 2
ADV 2
JSD_ADV 2
rm -rf runs/$logdir


python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ \
archives/$logdir/PS_alldata/ archives/$logdir/PS/ \
archives/$logdir/JSD/  archives/$logdir/ADV/ archives/$logdir/JSD_ADV/ --file val_dice.npy --axis 1 2 3 --postfix=model0 --seg_id=0 --y_lim 0.3 0.9

python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ \
archives/$logdir/PS_alldata/ archives/$logdir/PS/ \
archives/$logdir/JSD/  archives/$logdir/ADV/ archives/$logdir/JSD_ADV/  --file val_dice.npy --axis 1 2 3 --postfix=model1 --seg_id=1 --y_lim 0.3 0.9

Summary FS 2
Summary PS 2
#Summary PS_alldata 2
Summary JSD 2
Summary ADV 2
Summary JSD_ADV 2

python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv

zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir
