#!/bin/bash
set -e

echo "======================================"
echo " Starting Full Overnight Pipeline Run "
echo "======================================"

# 1. Build the data pipeline (Extract -> Clean -> Split -> Embed)
echo ">>> Running Data Pipeline..."
PYTHONPATH=. ./env/bin/python build_pipeline.py --dataset both

# 2. Build the BM25 Lexical Index
echo ">>> Building BM25 Index..."
PYTHONPATH=. ./env/bin/python src/retrieval/bm25.py

# 3. Evaluate Q2 (BM25 Recall)
echo ">>> Evaluating Lexical Recall@K (Q2)..."
PYTHONPATH=. ./env/bin/python src/retrieval/eval_bm25.py

# 4. Evaluate Q3 (Semantic Recall)
echo ">>> Evaluating Semantic Recall@K (Q3)..."
PYTHONPATH=. ./env/bin/python src/retrieval/eval_semantic.py

# 5. Execute Q4/Q5 Evaluation Harness
echo ">>> Executing Final Metrics Harness (AUC, MRR, nDCG, ILD)..."
PYTHONPATH=. ./env/bin/python src/evaluation/harness.py

echo "======================================"
echo " Overnight Run Complete! "
echo " Check data/feature_store/*/metrics_*.json for final numbers."
echo "======================================"
