#!/usr/bin/env bash
#!/usr/bin/env bash
num_model=$1
max_peoch=$2

set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

data_aug=None
net=enet
logdir=cardiac/task_5_multiple_view_${num_model}
tqdm=False

if [ $num_model = 2 ]; then
label="[[1,1],[1,1]]"
elif [ $num_model = 3 ]; then
label="[[1,1],[1,1],[1,1]]"
elif [ $num_model = 4 ]; then
label="[[1,1],[1,1],[1,1],[1,1]]"
else
echo "Do not support num_model = ${num_model}"
    exit
fi

source utils.sh
cd ..

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
Lab_Partitions.label=$label Lab_Partitions.partition_sets=1 Lab_Partitions.partition_overlap=1 \
Arch.name=$net Trainer.use_tqdm=$tqdm
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
Trainer.max_epoch=$max_peoch Dataset.augment=$data_aug \
StartTraining.train_adv=False StartTraining.train_jsd=False \
Lab_Partitions.label=$label Lab_Partitions.partition_sets=0.5 Lab_Partitions.partition_overlap=1 \
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
Lab_Partitions.label=$label Lab_Partitions.partition_sets=0.5 Lab_Partitions.partition_overlap=1 \
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
Lab_Partitions.label=$label Lab_Partitions.partition_sets=0.5 Lab_Partitions.partition_overlap=1 \
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
Lab_Partitions.label=$label Lab_Partitions.partition_sets=0.5 Lab_Partitions.partition_overlap=1 \
Arch.name=$net Trainer.use_tqdm=$tqdm
Summary $currentfoldername $gpu
rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir
}

rm -rf archives/$logdir
mkdir -p archives/$logdir
rm -rf runs/$logdir
mkdir -p runs/$logdir


FS 0 &
Partial 0 &
JSD 0 &
wait_script
ADV 0 &
wait_script
JSD_ADV 0 &
wait_script

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
