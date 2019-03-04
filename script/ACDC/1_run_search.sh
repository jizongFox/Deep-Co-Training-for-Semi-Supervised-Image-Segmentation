#!/usr/bin/env bash

wrapper(){
    la_ratio=$1
    max_epoch=$2
    step_size=$3
    hour=$4
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="task1_${la_ratio}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=32000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     1_run.sh  $la_ratio $max_epoch $step_size
}

wrapper 0.1 500 150 48

 wrapper 0.2 250 75 48

 wrapper 0.3 166 50 48

 wrapper 0.4 125 37 48

wrapper 0.5 100 30 48

 wrapper 0.6 83 25 48

 wrapper 0.7 71 21 48

 wrapper 0.8 63 18 48

 wrapper 0.9 55 16 48
