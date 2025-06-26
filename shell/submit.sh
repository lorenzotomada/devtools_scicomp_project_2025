#!/bin/bash

# Ranges over which we iterate
n_processes=(1 2)
matrix_sizes=(5 10 20)

last_dim="${matrix_sizes[-1]}"
last_nproc="${n_processes[-1]}"

CONFIG_FILE="experiments/config.yaml"

#rm -rf logs
echo "IMPORTANT: please remember to delete the logs folder before calling this script."

# Backup the original config
cp $CONFIG_FILE ${CONFIG_FILE}.bak

for dim in "${matrix_sizes[@]}"; do
  for n_p in "${n_processes[@]}"; do
    echo "------------------"

    sed -i "s/^dim: .*/dim: $dim/" $CONFIG_FILE
    sed -i "s/^n_processes: .*/n_processes: $n_processes/" $CONFIG_FILE
    sed -i "s/^plot: .*/plot: false/" $CONFIG_FILE
    echo "Running with size=$dim and n_processes=$n_p"

    echo "------------------"

    if [[ "$dim" == "$last_dim" && "$n_p" == "$last_nproc" ]]; then
      echo "Plotting the results in the logs folder..."
      sed -i "s/^plot: .*/plot: true/" $CONFIG_FILE
    fi

    python scripts/profiling_memory.py
  done
done

# Restore the original config file
mv ${CONFIG_FILE}.bak $CONFIG_FILE

echo "Experiment completed!"