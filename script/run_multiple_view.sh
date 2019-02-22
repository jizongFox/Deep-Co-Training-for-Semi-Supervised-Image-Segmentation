#!/bin/sh
model_num=$1
max_epoch=$2
hour=$3
echo "Running: ${model_num}"
module load python/3.6
source $HOME/torchenv36/bin/activate
module load scipy-stack
sbatch  --job-name=multipleview_${model_num} \
 --nodes=1  \
 --gres=gpu:1 \
 --cpus-per-task=6  \
 --mem=32000M \
 --time=0-${hour}:00 \
 --account=def-chdesa \
 --mail-user=jizong.peng.1@etsmtl.net \
 --mail-type=ALL   \
5_multiple_views.sh $model_num $max_epoch

