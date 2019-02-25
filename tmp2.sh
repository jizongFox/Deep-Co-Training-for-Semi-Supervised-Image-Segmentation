#!/usr/bin/env bash
folder_list=$(find archives/cardiac/IMPORTANTenet_VAT_classwiseNoise -mindepth 1 -maxdepth 1 -type d)
echo   $folder_list

for l in $folder_list
do
    python  Summary.py --input_dir=$l
done