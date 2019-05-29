#!/usr/bin/env bash

main_folder=./archives
subfolders=$(find archives/ -type d)
echo $subfolders
for dir in ${subfolders}
do
	python Summary.py --input_dir=$dir
done