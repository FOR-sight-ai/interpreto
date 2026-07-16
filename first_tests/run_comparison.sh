#!/bin/bash
#SBATCH --job-name=actual_vit_4
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.out
#SBATCH --error=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.err

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests


module load miniconda3/25.5.1/none-none
source $(conda info --base)/etc/profile.d/conda.sh
conda activate conda_env

echo "Launching the function"
python3 vit_comparison_captum_interpreto.py