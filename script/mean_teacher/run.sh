#!/usr/bin/env bash
wrapper_ACDC(){
    hour=$1
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="ACDC_meanteacher" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=10000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     run_mean_teacher_ACDC.sh
}

wrapper_GM(){
    hour=$1
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="GM_mean_teacher" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=10000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     run_mean_teacher_GM.sh
}
#wrapper_ACDC 20
#wrapper_GM 20
bash run_mean_teacher_GM.sh &
bash run_mean_teacher_ACDC.sh &