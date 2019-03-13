#!/usr/bin/env bash
seed=$1
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}


logdir=cardiac/$"Detailedresultsforoneparameteronconfig_seed_${seed}"

echo "Experiment Summary:"
source utils.sh


Summary(){
set -e
subfolder=$1
gpu=$2
echo CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/${logdir}/$subfolder
CUDA_VISIBLE_DEVICES=$gpu python Summary.py --input_dir runs/${logdir}/$subfolder
}

FS(){
set -e
gpu=$1
currentfoldername=FS
rm -rf runs/${logdir}/${currentfoldername}
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/${logdir}/${currentfoldername} \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.partition_sets=1 Lab_Partitions.partition_overlap=1 \
Trainer.use_tqdm=False Seed=${seed}
Summary ${currentfoldername} $gpu
rm -rf archives/${logdir}/${currentfoldername}
mv -f runs/${logdir}/${currentfoldername} archives/${logdir}
}

PS(){
set -e
gpu=$1
currentfoldername=PS
rm -rf runs/${logdir}/${currentfoldername}
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/${logdir}/${currentfoldername} \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Trainer.use_tqdm=False Seed=${seed}
Summary ${currentfoldername} $gpu
rm -rf archives/${logdir}/${currentfoldername}
mv -f runs/${logdir}/${currentfoldername} archives/${logdir}
}

JSD(){
set -e
gpu=$1
currentfoldername="JSD"
rm -rf runs/${logdir}/${currentfoldername}
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/${logdir}/${currentfoldername} \
StartTraining.train_adv=False StartTraining.train_jsd=True \
Trainer.use_tqdm=False Seed=${seed}
Summary ${currentfoldername} ${gpu}
rm -rf archives/${logdir}/${currentfoldername}
mv -f runs/${logdir}/${currentfoldername} archives/${logdir}
}


ADV(){
set -e
gpu=$1
currentfoldername="ADV"
rm -rf runs/${logdir}/${currentfoldername}
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/${logdir}/${currentfoldername} \
StartTraining.train_adv=True StartTraining.train_jsd=False \
Trainer.use_tqdm=False Seed=${seed}
Summary ${currentfoldername} $gpu
rm -rf archives/${logdir}/${currentfoldername}
mv -f runs/${logdir}/${currentfoldername} archives/${logdir}
}


JSD_ADV(){
set -e
gpu=$1
currentfoldername="JSD_ADV"
rm -rf runs/${logdir}/${currentfoldername}
CUDA_VISIBLE_DEVICES=$gpu python train_ACDC_cotraining.py Trainer.save_dir=runs/${logdir}/${currentfoldername} \
StartTraining.train_adv=True StartTraining.train_jsd=True \
Trainer.use_tqdm=False Seed=${seed}
Summary ${currentfoldername} $gpu
rm -rf archives/${logdir}/${currentfoldername}
mv -f runs/${logdir}/${currentfoldername} archives/${logdir}
}

cd ../..

mkdir -p archives/${logdir}
mkdir -p runs/${logdir}

FS 0 &

PS 0 &

JSD 0 &

JSD_ADV 0 &

ADV 0 &

wait_script



