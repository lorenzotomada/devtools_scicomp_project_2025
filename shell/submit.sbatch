#!/bin/bash

#SBATCH --partition=gpu2
#SBATCH --job-name=dtsc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10000
#SBATCH --time=12:00:00
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
export SLURM_NTASKS_PER_NODE=1  # due to Ulysses's bug

module load cuda/12.1
conda init
source ~/.bashrc
conda activate /scratch/ltomada/miniconda3/envs/dtsc

mkdir -p tmp_data experiments logs
touch tmp_data/timings.csv

export LINE_PROFILE=0

density=0.1
tol=0.0001
max_iter=1000


for dim in 100 500 1000 5000; do
    echo "d: $dim" # for readability
    echo "dim: $dim" > experiments/config.yaml    
    echo "density: $density" >> experiments/config.yaml
    echo "tol: $tol" >> experiments/config.yaml
    echo "max_iter: $max_iter" >> experiments/config.yaml

    python -m kernprof -l -o tmp_data/profile.dat scripts/run.py --config=experiments/config

    if [ ! -f tmp_data/timings.csv ]; then
        echo "dim,function_name,time" > tmp_data/timings.csv
    fi

    python -m line_profiler -rmt tmp_data/profile.dat | awk -v dim="$dim" '
        /Function: / {func_name=$2}
        /Total time:/ {if (func_name != "") print dim, func_name, $3}' OFS=',' >> tmp_data/timings.csv

    echo "dim: $dim - times saved in tmp_data/timings.csv" 
    echo '--------------------------------------------------'
done


python scripts/plot_scalability.py
echo "Experiment completed!"
#rm -rf tmp_data
