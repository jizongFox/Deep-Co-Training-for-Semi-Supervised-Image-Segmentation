#!/usr/bin/env bash
time=$(date +'%m%d_%H:%M')
gitcommit_number=$(git rev-parse HEAD)
gitcommit_number=${gitcommit_number:0:8}

la_ratio=0.5
max_epoch=100

logdir="task1_labeled_unlabeled_ratio_${la_ratio}"

bash 1_labeled_unlabeled_ratio.sh $logdir FS $max_epoch $la_ratio 1

bash 1_labeled_unlabeled_ratio.sh $logdir PS $max_epoch $la_ratio 1

bash 1_labeled_unlabeled_ratio.sh $logdir ADV $max_epoch $la_ratio 1

bash 1_labeled_unlabeled_ratio.sh $logdir JSD $max_epoch $la_ratio 1

bash 1_labeled_unlabeled_ratio.sh $logdir JSD_ADV $max_epoch $la_ratio 1

python generalframework/postprocessing/report.py --folder=archives/$logdir/ --file=summary.csv