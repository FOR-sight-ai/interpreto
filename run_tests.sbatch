#!/bin/bash
#SBATCH --job-name=pytests
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.out
#SBATCH --error=/gpfs/users/debosschu/interpretability_libraries/interpreto/first_tests/logs/%j.err

cd /gpfs/users/debosschu/interpretability_libraries/interpreto


source /gpfs/users/debosschu/interpretability_libraries/interpreto/.new_venv/bin/activate
echo "Launching the function"
python3 -m pytest -v -ra --tb=short --color=no tests/model_wrapping/test_image_classification_inference_wrapper.py
