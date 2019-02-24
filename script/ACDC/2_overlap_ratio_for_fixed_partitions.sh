#!/usr/bin/env bash
set -e
logdir=$1
received_com=$2
max_peoch=$3
overlap_ratio=$4
labeled_unlabeled_ratio=$5
data_aug=None
net=enet
tqdm=False
source utils.sh

cd ../..
Summary(){
set -e
subfolder=$1
gpu=$2
echo CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/$logdir/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/$logdir/$subfolder
}

FS(){
set -e

gpu=$1
currentfoldername=FS
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.partition_sets=1 Lab_Partitions.partition_overlap=$overlap_ratio \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

PS(){
set -e

gpu=$1
currentfoldername=PS
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$overlap_ratio \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD(){
set -e

gpu=$1
currentfoldername=JSD
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=True \
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$overlap_ratio \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

ADV(){
set -e
gpu=$1
currentfoldername=ADV
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=False \
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$overlap_ratio \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

JSD_ADV(){
set -e
gpu=$1
currentfoldername=JSD_ADV
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$overlap_ratio \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}
mkdir -p archives/$logdir
mkdir -p runs/$logdir
echo $received_com
$received_com 0
#wait_script