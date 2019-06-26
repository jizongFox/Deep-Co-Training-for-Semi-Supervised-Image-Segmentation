#!/usr/bin/env bash
folder_list=$(find archives/cardiac/FSGM_new_loss_3_13 -mindepth 2 -maxdepth 2 -type d)
echo   $folder_list
#
#for f in $folder_list:
#do
#	python Summary.py --input_dir=$f
#done
python generalframework/postprocessing/report.py --folder=archives/cardiac/FSGM_new_loss_3_13 --file=bsummary.csv