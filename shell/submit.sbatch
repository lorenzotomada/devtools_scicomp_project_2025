#!/bin/bash
# SLURM job options
#SBATCH --partition=gpu2
#SBATCH --job-name=dtsc_alexnet
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=30000
#SBATCH --time=00:10:00
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
# ==== End of Info part ===== #
#

cd $SLURM_SUBMIT_DIR # probably not necessary with SLURM, but better to include it
export SLURM_NTASKS_PER_NODE=2  # Ulysses has a bug :( need to inlude this

module load cuda/12.1
conda activate devtools_scicomp
srun python scripts/run.py fit --config=experiments/config.yaml --trainer.strategy=ddp --trainer.devices=2
