#!/usr/bin/env bash
set -e
cd ..
max_peoch=10
data_aug=None
net=enet
logdir=cardiac/$net"_VAT"
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
Dataset.augment=$data_aug  StartTraining.train_adv=False  Arch.name=$net Lab_Partitions.label=[[1,61]] Lab_Partitions.unlabel=[61,101]
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir

}

ADV(){
currentfoldername=ADV
gpu=$1
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_vat.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_adv=True  Arch.name=$net Lab_Partitions.label=[[1,61]] Lab_Partitions.unlabel=[61,101]
Summary  $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

rm -rf archives/$logdir
mkdir -p archives/$logdir
rm -rf runs/$logdir

FS 0 &
PS 0 &
ADV 0 &
wait_script

python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ archives/$logdir/PS/ archives/$logdir/ADV/  --file val_dice.npy --axis 1 2 3 --postfix=test --seg_id=0
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS/ archives/$logdir/FS/ archives/$logdir/ADV/  --file val_batch_dice.npy --axis 1 2 3 --postfix=test --seg_id=0

python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv
rm -rf runs/$logdir