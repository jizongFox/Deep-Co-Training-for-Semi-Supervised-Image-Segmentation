#!/usr/bin/env bash
set -e
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

num_models=$1
max_epoch=$2
gpu=$3
seed=$4

logdir="multiview_${num_models}_seed_${seed}"
source ./utils.sh
if [ $num_models = 2 ]; then

	bash 5_multiple_views.sh $logdir FS $num_models $max_epoch $gpu $seed &
	bash 5_multiple_views.sh $logdir JSD $num_models $max_epoch $gpu $seed &

	wait_script
	bash 5_multiple_views.sh $logdir PS $num_models $max_epoch $gpu $seed &
	bash 5_multiple_views.sh $logdir JSD_ADV $num_models $max_epoch $gpu $seed &
	wait_script
fi

if [ $num_models = 3 ]; then

    bash 5_multiple_views.sh $logdir PS $num_models $max_epoch $gpu $seed &
	wait_script

	bash 5_multiple_views.sh $logdir JSD $num_models $max_epoch $gpu $seed &
	wait_script

	bash 5_multiple_views.sh $logdir ADV $num_models $max_epoch $gpu $seed &
	wait_script
	bash 5_multiple_views.sh $logdir JSD_ADV $num_models $max_epoch $gpu $seed &
	wait_script
fi

if [ $num_models = 4 ]; then

    bash 5_multiple_views.sh $logdir PS $num_models $max_epoch $gpu $seed &
	wait_script

	bash 5_multiple_views.sh $logdir JSD $num_models $max_epoch $gpu $seed &
	wait_script

    bash 5_multiple_views.sh $logdir ADV $num_models $max_epoch $gpu $seed &
	wait_script
	bash 5_multiple_views.sh $logdir JSD_ADV $num_models $max_epoch $gpu $seed &
	wait_script
fi



cd ../..
python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=bsummary.csv

zip -rq archives/$logdir"_"$time"_"$gitcommit_number".zip" archives/$logdir