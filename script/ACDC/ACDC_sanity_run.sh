#!/usr/bin/env bash


wrapper(){
    seed=$1
    hour=$2
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="getting detailed results" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=10000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    ACDC_sanity_check.sh $seed
}

wrapper 1234 1
wrapper 1235 1