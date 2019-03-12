#!/usr/bin/env bash
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

la_ratio=$1
max_epoch=$2
overlap_ratio=1

source utils.sh
logdir="cardiac/task1_labeled_unlabeled_ratio_${la_ratio}"

bash 1_labeled_unlabeled_ratio.sh $logdir FS $max_epoch $la_ratio $overlap_ratio  &
bash 1_labeled_unlabeled_ratio.sh $logdir JSD_ADV $max_epoch $la_ratio $overlap_ratio &
bash 1_labeled_unlabeled_ratio.sh $logdir PS $max_epoch $la_ratio $overlap_ratio  &
bash 1_labeled_unlabeled_ratio.sh $logdir JSD $max_epoch $la_ratio $overlap_ratio &
bash 1_labeled_unlabeled_ratio.sh $logdir ADV $max_epoch $la_ratio $overlap_ratio &
wait_script
cd ../..
python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=bsummary.csv
zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir
