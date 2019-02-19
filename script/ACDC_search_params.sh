#!/usr/bin/env bash
groupname=$1

set -e
cd ..
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

max_epoch=$2
data_aug=None
net=enet
logdir=cardiac/$net"_search"$groupname
FAIL=0

wait_script(){
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi
}

Summary(){
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
Trainer.max_epoch=$max_epoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,101],[1,101]]" \
Arch.name=$net Trainer.use_tqdm=False
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

Partial(){
set -e
gpu=$1
currentfoldername=PS
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_epoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" \
Arch.name=$net Trainer.use_tqdm=False
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}


ADV(){
set -e
gpu=$1
Cot_beg_epoch=$2
Cot_max_value=$3
Cot_max_epoch=$4
Adv_beg_epoch=$5
Adv_max_value=$6
Adv_max_epoch=$7
currentfoldername="ADV"$Cot_max_value""$Cot_max_epoch""$Cot_beg_epoch""$Adv_max_value""$Adv_max_epoch""$Adv_beg_epoch
ramp_mult=-5
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_epoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" Arch.name=$net \
Cot_Scheduler.begin_epoch=$Cot_beg_epoch Cot_Scheduler.max_epoch=$Cot_max_epoch Cot_Scheduler.max_value=$Cot_max_value \
Cot_Scheduler.ramp_mult=$ramp_mult \
Adv_Scheduler.begin_epoch=$Adv_beg_epoch Adv_Scheduler.max_epoch=$Adv_max_epoch Adv_Scheduler.max_value=$Adv_max_value \
Adv_Scheduler.ramp_mult=$ramp_mult \
Trainer.use_tqdm=False
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}


JSD_ADV(){
set -e
gpu=$1
Cot_beg_epoch=$2
Cot_max_value=$3
Cot_max_epoch=$4
Adv_beg_epoch=$5
Adv_max_value=$6
Adv_max_epoch=$7
currentfoldername="JSD_ADV"$Cot_max_value""$Cot_max_epoch""$Cot_beg_epoch""$Adv_max_value""$Adv_max_epoch""$Adv_beg_epoch
ramp_mult=-5
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_epoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Lab_Partitions.label="[[1,61],[1,61]]" Lab_Partitions.unlabel="[61,101]" Arch.name=$net \
Cot_Scheduler.begin_epoch=$Cot_beg_epoch Cot_Scheduler.max_epoch=$Cot_max_epoch Cot_Scheduler.max_value=$Cot_max_value \
Cot_Scheduler.ramp_mult=$ramp_mult \
Adv_Scheduler.begin_epoch=$Adv_beg_epoch Adv_Scheduler.max_epoch=$Adv_max_epoch Adv_Scheduler.max_value=$Adv_max_value \
Adv_Scheduler.ramp_mult=$ramp_mult \
Trainer.use_tqdm=False
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

#rm -rf archives/$logdir
mkdir -p archives/$logdir
#rm -rf runs/$logdir
mkdir -p runs/$logdir

group1(){
FS 0 &
Partial 0
}

#
group2(){
ADV 0 0 1 80 10 0.001 80  &
JSD_ADV 0 0 1 80 10 0.001 80
}

group3(){
ADV 0 0 1 80 10 0.01 80 &
JSD_ADV 0 0 1 80 10 0.01 80
}

group4(){
ADV 0 0 1 80 10 0.1 80 &
JSD_ADV 0 0 1 80 10 0.1 80
}

group5(){
ADV 0 0 1 80 10 0.5 80 &
JSD_ADV 0 0 1 80 10 0.5 80
}

group6(){
ADV 0 0 1 80 10 0.8 80 &
JSD_ADV 0 0 1 80 10 0.8 80
}
group7(){
ADV 0 0 1 80 10 1 80 &
JSD_ADV 0 0 1 80 10 1 80
}

## execute
echo $groupname
$groupname
wait_script


#zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir
#rm -rf runs/$logdir
