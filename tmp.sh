#!/usr/bin/env bash
listfolder=$(find  /home/jizong/Desktop/compare/ -mindepth 2 -maxdepth 2 -type d )
#for l in $listfolder
#do
#python Summary.py --input_dir=$l
#done
listfolder=$(find  /home/jizong/Desktop/compare/ -mindepth 1 -maxdepth 1 -type d )
for l in $listfolder
do
python generalframework/postprocessing/report.py --folder=$l --file=summary.csv
done
