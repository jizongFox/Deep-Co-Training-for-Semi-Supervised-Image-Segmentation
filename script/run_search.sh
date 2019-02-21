#!/bin/sh
jobname=$1
max_epoch=$2
echo "Running: ${jobname}"
module load python/3.6
source $HOME/torchenv36/bin/activate
module load scipy-stack
sbatch  --job-name=$jobname \
 --nodes=1  \
 --gres=gpu:1 \
 --cpus-per-task=6  \
 --mem=32000M \
 --time=0-19:30 \
 --account=def-chdesa \
 --mail-user=jizong.peng.1@etsmtl.net \
 --mail-type=ALL   \
 ACDC_search_params.sh $jobname $max_epoch
