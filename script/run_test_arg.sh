#!/bin/sh
jobname=$1
max_epoch=$2
#SBATCH --job-name="acdc_search_"${jobname}
#SBATCH --time=0-00:03
#SBATCH --account=def-chdesa
echo "Job Name: "${jobname}
echo "Running ${jobname} for ${max_epoch} epochs!"
