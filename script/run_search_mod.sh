#!/bin/sh
jobname=$1
max_epoch=$2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="acdc_search_"${jobname}
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M
#SBATCH --time=0-00:30
#SBATCH --account=def-chdesa
#SBATCH --mail-user=guilled52@gmail.com
#SBATCH --mail-type=ALL
echo "Running: ${jobname}"
module load python/3.6
source $HOME/torchenv36/bin/activate
module load scipy-stack
time bash ACDC_search_params.sh $jobname

