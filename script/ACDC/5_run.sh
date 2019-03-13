#!/usr/bin/env bash
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

num_models=$1
max_epoch=$2
logdir="multiview_${num_models}"
source ./utils.sh

bash 5_multiple_views.sh $logdir FS $num_models $max_epoch
bash 5_multiple_views.sh $logdir PS $num_models $max_epoch
bash 5_multiple_views.sh $logdir ADV $num_models $max_epoch
bash 5_multiple_views.sh $logdir JSD $num_models $max_epoch
bash 5_multiple_views.sh $logdir JSD_ADV $num_models $max_epoch
wait_script

cd ../..
python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv

zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir