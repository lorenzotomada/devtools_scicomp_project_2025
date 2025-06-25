#!/bin/bash

module load cmake/3.29.1
module load intel/2021.2
module load openmpi3/3.1.4

conda activate devtools_scicomp

# Ranges over which we iterate
n_processes=(1 2)
matrix_sizes=(10 50 100 500)

CONFIG_FILE="experiments/config.yaml"


# Backup the original config
cp $CONFIG_FILE ${CONFIG_FILE}.bak

for dim in "${matrix_sizes[@]}"; do
  for n_p in "${n_processes[@]}"; do
    sed -i "s/^dim: .*/dim: $dim/" $CONFIG_FILE
    sed -i "s/^n_processes: .*/n_processes: $n_processes/" $CONFIG_FILE
    echo "Running with size=$dim and n_processes=$n_p"
    mpirun -np 1 python scripts/profiling_memory.py
  done
done

# Restore the original config file
mv ${CONFIG_FILE}.bak $CONFIG_FILE

#python scripts/plot_scalability.py
echo "Experiment completed!"
