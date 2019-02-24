#!/usr/bin/env bash
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

la_ratio=0.5
overlap_ratio=1
max_epoch=1

source utils.sh
logdir="cardiac/task2_overlap_ratio_${overlap_ratio}_for_${la_ratio}_la_ratio"
echo $logdir

bash 2_overlap_ratio_for_fixed_partitions.sh $logdir FS $max_epoch $overlap_ratio $la_ratio &
bash 2_overlap_ratio_for_fixed_partitions.sh $logdir PS $max_epoch $overlap_ratio $la_ratio &
wait_script
bash 2_overlap_ratio_for_fixed_partitions.sh $logdir ADV $max_epoch $overlap_ratio $la_ratio &
bash 2_overlap_ratio_for_fixed_partitions.sh $logdir JSD $max_epoch $overlap_ratio $la_ratio &
bash 2_overlap_ratio_for_fixed_partitions.sh $logdir JSD_ADV $max_epoch $overlap_ratio $la_ratio &
wait_script


cd ../..
python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv

zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir