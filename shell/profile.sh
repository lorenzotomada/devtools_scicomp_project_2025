#!/bin/bash

# Ranges over which we iterate
n_processes=(1 2) # 4 8 16)
matrix_sizes=(10 50 100 500 1000)

CONFIG_FILE="experiments/config_profiling.yaml"

# Backup the original config
cp $CONFIG_FILE ${CONFIG_FILE}.bak

for dim in "${matrix_sizes[@]}"; do
  sed -i "s/^dim: .*/dim: $dim/" $CONFIG_FILE

  for np in "${n_processes[@]}"; do
    echo "Running with size=$dim and n_processes=$np"
    mpirun -np $np python scripts/profiling.py
  done
done

# Restore the original config file
mv ${CONFIG_FILE}.bak $CONFIG_FILE
