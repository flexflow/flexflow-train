#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=SbatchTest
#SBATCH --output=sbatch_test.txt
#SBATCH --error=sbatch_test.err
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4138

set -x
set -e

echo "Current directory: $PWD"

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

echo "Directory after cd: $PWD"

LOG_FILE="./sbatch_test.log"

echo "Writing to log file now" 2>&1 | tee $LOG_FILE
