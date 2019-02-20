#!/usr/bin/env bash
set -e
cd ..
max_peoch=3
data_aug=None
net=enet
use_tqdm=False
logdir=cardiac/$net"_VAT"_epsilions_scheduler
mkdir -p archives/$logdir

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
ADV(){

gpu=$1
lmax=$2
currentfoldername=adv_{$lmax}
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True  Arch.name=$net Lab_Partitions.label=[[1,61]] Lab_Partitions.unlabel=[61,101] \
Adv_Scheduler.max_value=$lmax StartTraining.use_tqdm=$use_tqdm
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}


rm -rf archives/$logdir
mkdir -p archives/$logdir
rm -rf runs/$logdir
ADV 1 0.001 &
ADV 2 0.005 &
wait_script

