#!/bin/bash

#SBATCH --partition=regular1
#SBATCH --job-name=dtsc
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --mail-user=ltomada@sissa.it
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N

# Print job details
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
#
# ==== End of Info part (say things) ===== #
#

cd $SLURM_SUBMIT_DIR

module load cmake/3.29.1
module load intel/2021.2
module load openmpi3/3.1.4

conda init
conda activate devtools_scicomp

# Ranges over which we iterate
n_processes=(2 4 8)
matrix_sizes=(10 50 100 500 1000)

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

    srun --mpi=openmpi -n ${n_p} python scripts/profiling_memory_and_time.py
  done
done

# Restore the original config file
mv ${CONFIG_FILE}.bak $CONFIG_FILE

echo "Experiment completed!"
