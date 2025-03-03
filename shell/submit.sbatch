#!/bin/bash

#SBATCH --partition=gpu2
#SBATCH --job-name=dtsc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32000
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
export SLURM_NTASKS_PER_NODE=2  # due to Ulysses's bug

module load cuda/12.1
conda init
conda activate ~/miniconda3/envs/devtools_scicomp

# Run the script
python scripts/run.py fit --config=experiments/config.yaml
