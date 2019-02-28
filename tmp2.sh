#!/usr/bin/env bash
folder_list=$(find /home/jizong/Desktop/task1/archives/cardiac -mindepth 1 -maxdepth 1 -type d)
echo   $folder_list

#for l in $folder_list
#do
#    python  Summary.py --input_dir=$l
#done

for l in $folder_list
do
python generalframework/postprocessing/report.py --folder=$l --file=summary.csv
done
