#!/bin/bash
#SBATCH --job-name=actual_vit
#SBATCH --partition=mem_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.out
#SBATCH --error=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.err

cd /gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests

source /gpfs/users/debosschu/interpretability_libraries/interpreto/.new_venv/bin/activate

python3 actual_vit.py
