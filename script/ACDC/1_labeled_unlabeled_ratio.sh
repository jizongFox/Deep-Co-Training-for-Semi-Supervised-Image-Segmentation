#!/usr/bin/env bash
set -e
# input parameters
logdir=$1
received_com=$2
max_peoch=$3
labeled_unlabeled_ratio=$4
partiton_overlap=$5
step_size=$6
echo "step_size: ${step_size}"
data_aug=None
net=enet
tqdm=False
random_seed=2
echo "random seed: ${random_seed}"
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
Trainer.max_epoch=100 Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.partition_sets=1 Lab_Partitions.partition_overlap=$partiton_overlap \
Arch.name=$net Trainer.use_tqdm=$tqdm Seed=$random_seed
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
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$partiton_overlap \
Arch.name=$net Trainer.use_tqdm=$tqdm Seed=$random_seed Scheduler.step_size=$step_size
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
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$partiton_overlap \
Arch.name=$net Trainer.use_tqdm=$tqdm Seed=$random_seed Scheduler.step_size=$step_size
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
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$partiton_overlap \
Arch.name=$net Trainer.use_tqdm=$tqdm Seed=$random_seed Scheduler.step_size=$step_size
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
Lab_Partitions.partition_sets=$labeled_unlabeled_ratio Lab_Partitions.partition_overlap=$partiton_overlap \
Arch.name=$net Trainer.use_tqdm=$tqdm Seed=$random_seed Scheduler.step_size=$step_size
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

mkdir -p archives/$logdir
mkdir -p runs/$logdir

echo $received_com
$received_com 0


#python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ \
# archives/$logdir/PS/ \
#archives/$logdir/JSD/  archives/$logdir/ADV/ archives/$logdir/JSD_ADV/ --file val_dice.npy --axis 1 2 3 --postfix=model0 --seg_id=0 --y_lim 0.3 0.9
#
#python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ \
# archives/$logdir/PS/ \
#archives/$logdir/JSD/  archives/$logdir/ADV/ archives/$logdir/JSD_ADV/  --file val_dice.npy --axis 1 2 3 --postfix=model1 --seg_id=1 --y_lim 0.3 0.9
#
#python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv
#
#zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir
#rm -rf runs/$logdir
