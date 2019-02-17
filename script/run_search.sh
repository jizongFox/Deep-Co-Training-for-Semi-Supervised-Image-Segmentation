#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="acdc_search"
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=0-36:00
#SBATCH --account=def-chdesa
#SBATCH --mail-user=jizong.peng.1@etsmtl.net
#SBATCH --mail-type=ALL
jobname=$1
echo $jobname
sbatch  --job-name=$jobname  --nodes=1  --gres=gpu:1  --cpus-per-task=12   --mem=32000M  --time=0-36:00 --account=def-chdesa --mail-user=jizong.peng.1@etsmtl.net --mail-type=ALL   ACDC_search_params.sh $jobname
