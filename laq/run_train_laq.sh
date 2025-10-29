#!/bin/bash
#SBATCH --job-name=laq1
#SBATCH --output=jobs/laq1.%j.out
#SBATCH --error=jobs/laq1.%j.err
#SBATCH --partition=dev
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00

set -euo pipefail

source /fsx/sroutray/miniconda3/etc/profile.d/conda.sh
conda activate laq

accelerate launch --num_processes=8 --mixed_precision=bf16 --main_process_port=27562 train_laq.py