#!/bin/bash

# Slurm Options
#SBATCH -o other.out-%j
#SBATCH -c 32
#SBATCH --exclusive

# Initialize module source
source /etc/profile

module add julia

./run_job.sh

# To submit this job: `sbatch log_execution_times.sh`