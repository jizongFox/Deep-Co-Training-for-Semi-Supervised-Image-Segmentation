#!/usr/bin/env bash

 #rrg-mpederso, def-mpederso, and def-chdesa
bash run.sh 0.05 256 rrg-mpederso
bash run.sh 0.10 256 rrg-mpederso
bash run.sh 0.20 256 rrg-mpederso

bash run.sh 0.05 128 def-mpederso
bash run.sh 0.10 128 def-mpederso
bash run.sh 0.20 128 def-mpederso

bash run.sh 0.05 512 def-chdesa
bash run.sh 0.10 512 def-chdesa
bash run.sh 0.20 512 def-chdesa