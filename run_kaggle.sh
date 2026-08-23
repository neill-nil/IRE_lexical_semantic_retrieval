#!/bin/bash
set -e

echo "======================================"
echo " Preparing Kaggle Environment "
echo "======================================"

pip install -r requirements.txt

echo "======================================"
echo " Make sure your datasets are mounted! "
echo " Update generate_submission.py with the exact Kaggle Dataset paths."
echo "======================================"

# PYTHONPATH=. python build_pipeline.py --dataset mind
# PYTHONPATH=. python src/evaluation/generate_submission.py
