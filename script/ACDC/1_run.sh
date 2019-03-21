#!/usr/bin/env bash
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

la_ratio=$1
max_epoch=$2
seed=$3
gpu=$4

source utils.sh
logdir="cardiac/task1_labeled_unlabeled_ratio_${la_ratio}_seed_${seed}"

bash 1_labeled_unlabeled_ratio.sh $logdir PS $max_epoch $la_ratio $gpu $seed &
bash 1_labeled_unlabeled_ratio.sh $logdir FS $max_epoch $la_ratio $gpu $seed &
bash 1_labeled_unlabeled_ratio.sh $logdir JSD $max_epoch $la_ratio $gpu $seed &
wait_script
bash 1_labeled_unlabeled_ratio.sh $logdir ADV $max_epoch $la_ratio $gpu $seed &
wait_script
bash 1_labeled_unlabeled_ratio.sh $logdir JSD_ADV $max_epoch $la_ratio $gpu $seed &
wait_script
cd ../..
python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=bsummary.csv
zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir
