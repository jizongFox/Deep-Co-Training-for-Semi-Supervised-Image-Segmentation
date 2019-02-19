#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="acdc_search_"${jobname}
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M
#SBATCH --time=0-24:00
#SBATCH --account=def-chdesa
#SBATCH --mail-user=jizong.peng.1@etsmtl.net
#SBATCH --mail-type=ALL
jobname=$1
max_epoch=$2
echo "Running: ${jobname}"
module load python/3.6
source $HOME/torchenv36/bin/activate
module load scipy-stack
bash ACDC_search_params.sh $jobname $max_epoch

