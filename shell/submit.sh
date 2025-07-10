#!/bin/bash

# Ranges over which we iterate
n_processes=(1 2 4 8)
matrix_sizes=(10 50 100 500 1000 1500)

last_dim="${matrix_sizes[-1]}"
last_nproc="${n_processes[-1]}"

CONFIG_FILE="experiments/config.yaml"

mkdir -p logs
rm logs/*
mkdir -p Profiling_files
rm Profiling_files/*

# Backup the original config
cp $CONFIG_FILE ${CONFIG_FILE}.bak

for dim in "${matrix_sizes[@]}"; do
  for n_p in "${n_processes[@]}"; do
    echo "------------------"

    sed -i "s/^dim: .*/dim: $dim/" $CONFIG_FILE
    sed -i "s/^plot: .*/plot: false/" $CONFIG_FILE
    echo "Running with size=$dim and n_processes=$n_p"

    echo "------------------"

    if [[ "$dim" == "$last_dim" && "$n_p" == "$last_nproc" ]]; then
      echo "Plotting the results in the logs folder..."
      sed -i "s/^plot: .*/plot: true/" $CONFIG_FILE
    fi

    mpirun -np ${n_p} python scripts/profiling_memory_and_time.py
  done
done

# Restore the original config file
mv ${CONFIG_FILE}.bak $CONFIG_FILE

echo "Experiment completed!"
