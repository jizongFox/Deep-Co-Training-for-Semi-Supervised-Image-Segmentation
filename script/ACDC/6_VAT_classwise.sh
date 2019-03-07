#!/usr/bin/env bash
set -e
cd ..
max_peoch=$1
data_aug=None
net=enet
use_tqdm=False
logdir=cardiac/$net"_VAT"_classwiseNoise
mkdir -p archives/$logdir


wait_script(){
FAIL=0
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
echo CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/${logdir}/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/${logdir}/$subfolder
}

FS(){
currentfoldername=FS
gpu=$1
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False  Arch.name=$net Lab_Partitions.label=[[1,101]]
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

PS(){
currentfoldername=PS
gpu=$1
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug  StartTraining.train_adv=False  Arch.name=$net Lab_Partitions.label=[[1,51]]
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

ADV(){
set -e
gpu=$1
lmax=$2
axises=$3
currentfoldername=adv_${lmax}_${axises}
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True  Arch.name=$net Lab_Partitions.label=[[1,51]] Lab_Partitions.unlabel=[51,101] \
Adv_Scheduler.max_value=$lmax StartTraining.use_tqdm=$use_tqdm  Adv_Training.vat_axises=$axises
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

#rm -rf archives/$logdir
mkdir -p archives/$logdir
#rm -rf runs/$logdir
FS 0 &
ADV 0 0.01 [1] &
ADV 0 0.01 [2] &
ADV 0 0.01 [3] &
ADV 0 0.01 [0] &
ADV 0 0.01 [1,2,3] &
ADV 0 0.01 [0,1,2,3] &
PS 0
wait_script

python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv

