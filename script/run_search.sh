#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="acdc_search"
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M
#SBATCH --time=0-35:30
#SBATCH --account=def-chdesa
#SBATCH --mail-user=jizong.peng.1@etsmtl.net
#SBATCH --mail-type=ALL
jobname=$1
sbatch  --job-name=$jobname ACDC_search_params.sh $jobname