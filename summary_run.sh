#!/usr/bin/env bash

subfolders=$(find runs/spleen_re_512/0.10 -type d)
echo $subfolders
for dir in ${subfolders}
do
	python Summary.py --input_dir=$dir
done