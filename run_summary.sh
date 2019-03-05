#!/usr/bin/env bash
folder_list=$(find /home/jizong/Desktop/new_results/ -mindepth 4 -maxdepth 4 -type d)
echo   $folder_list

for f in $folder_list:
do
	python Summary.py --input_dir=$f
done
python generalframework/postprocessing/report.py --folder=/home/jizong/Desktop/task1_labeled_unlabeled_ratio_0.5/ --file=summary.csv