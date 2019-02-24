#!/usr/bin/env bash


wrapper(){
    overlap_ratio=$1
    max_epoch=$2
    hour=$3
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="task2_${overlap_ratio}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=32000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     2_run.sh  $overlap_ratio $max_epoch
}

wrapper 0 100 72
wrapper 0.2 100 72
wrapper 0.4 100 72
wrapper 0.6 100 72
wrapper 0.8 100 72
wrapper 1 100 72