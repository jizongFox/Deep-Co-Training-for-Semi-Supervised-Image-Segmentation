#!/usr/bin/env bash
source utils.sh

# adv = 0
bash check.sh 0 0 &
wait_script


# adv = 0.001
bash check.sh 0 0.001 &

wait_script
# adv = 0.005
bash check.sh 0 0.005 &

wait_script

# adv = 0.01
bash check.sh 0 0.01 &

wait_script

# adv = 0.05
bash check.sh 0 0.05 &
wait_script

# adv = 0.1
bash check.sh 0 0.1 &
wait_script
