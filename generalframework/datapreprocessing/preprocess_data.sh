#!/usr/bin/env bash
set -e
echo "-> Preprocessing the spin dataset"
# clear unziped folder
echo "-> unzip dataset to folders"
rm -rf data/train data/test data/unlabel
unzip -q  data/training-data-gm-sc-challenge-ismrm16-v20160302b.zip -d data/train
unzip -q  data/test-data-gm-sc-challenge-ismrm16-v20160401.zip -d data/unlabel
echo "-> Unzip dataset done"
rm -rf data/*/*.txt

rm -rf Slices
python slice_spin.py main data Slices




