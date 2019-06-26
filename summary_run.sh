#!/usr/bin/env bash

main_folder=./archives
subfolders=$(find archives/cardiac/multiview/local/New_Multiview/ -type d)
echo $subfolders
for dir in ${subfolders}
do
	python Summary.py --input_dir=$dir
done