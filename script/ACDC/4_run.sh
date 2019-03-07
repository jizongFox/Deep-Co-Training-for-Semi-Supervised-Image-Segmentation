#!/usr/bin/env bash


wrapper(){
    group=$1
    max_epoch=$2
    hour=$3
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="reserach_${group}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=10000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    4_parameter_search_adv_jsd.sh $group $max_epoch
}

wrapper group1 100 35
wrapper group2 100 35
wrapper group3 100 35
wrapper group4 100 35
wrapper group5 100 35
wrapper group6 100 35
wrapper group7 100 35