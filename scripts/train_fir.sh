#!/bin/bash
#SBATCH --account=def-six
#SBATCH --job-name=DNF_train
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=6:00:00
#SBATCH --mail-user=er.liang@mail.utoronto.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +"%Y-%m-%d/%H-%M-%S")
LOG_DIR="${SCRATCH}/dnf/logs/${TIMESTAMP}"
export WANDB_DIR="${SCRATCH}/dnf/wandb"

mkdir -p ${LOG_DIR}

exec > ${LOG_DIR}/slurm-${SLURM_JOB_ID}.out 2> ${LOG_DIR}/slurm-${SLURM_JOB_ID}.err

module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

source ENV/bin/activate

# --- DIAGNOSTICS ---
echo "--- Job Diagnostics ---"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Assigned GPUs: ${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES: '${CUDA_VISIBLE_DEVICES}'"
echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available in PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "--- nvidia-smi output ---"
nvidia-smi
echo "-------------------------"
# --- END DIAGNOSTICS ---

# Run the training script, overriding the Hydra run directory
python -m src.train \
    hydra.run.dir=${LOG_DIR} \
    data.dataset.num_workers=${SLURM_CPUS_PER_TASK} \
    "$@" # Pass command line arguments to the script