#!/usr/bin/env bash
set -e
max_epoch=$1
logdir=cardiac/task3_VAT_vs_FSGM
tqdm=False
data_aug=None
net=enet

source utils.sh
cd ..

Summary(){
set -e
subfolder=$1
gpu=$2
echo CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/$logdir/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/$logdir/$subfolder
}

JSD_ADV(){
set -e
gpu=$1
fsgm_ratio=$2
use_fsgm=$3
currentfoldername=$4
rm -rf runs/$logdir/$currentfoldername
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername \
Trainer.max_epoch=$max_epoch Dataset.augment=$data_aug \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Lab_Partitions.label="[[1,1],[1,1]]" Lab_Partitions.partition_sets=0.5 Lab_Partitions.partition_overlap=1 \
Arch.name=$net Trainer.use_tqdm=$tqdm Adv_Training.label_data_ratio=$fsgm_ratio Adv_Training.use_fsgm=$use_fsgm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

rm -rf archives/$logdir
mkdir -p archives/$logdir
rm -rf runs/$logdir

#JSD_ADV 0 0 &
##JSD_ADV 2 1 &
##JSD_ADV 1 0.2 &
##JSD_ADV 2 0.4 &
##JSD_ADV 1 0.6 &
##JSD_ADV 2 0.8 &
# use only labeled data with FSGM
JSD_ADV 0 1 True onlyuselabeleddata_FSGM &

# use only labeled data with VAT
JSD_ADV 0 1 False onlyuselabeleddata_VAT &

# use only unlabeled data
JSD_ADV 0 0 False onlyuseunlabeleddata &
# use mixed data FSGM+VAT
JSD_ADV 0 0.5 True FSGM_VAT &

# use mixed data VAT+VAT
JSD_ADV 0 0.5 False VAT_VAT &

wait_script

python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv
wait_script
rm -rf runs/$logdir
