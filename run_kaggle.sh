#!/bin/bash
set -e

echo "======================================"
echo " Preparing Kaggle Environment "
echo "======================================"

pip install -q -r requirements.txt

echo "======================================"
echo " Linking Kaggle-mounted datasets into data/raw/ "
echo " Edit the paths below to match what you see under /kaggle/input/ "
echo "======================================"

mkdir -p data/raw

# --- EDIT THESE to match your actual /kaggle/input/<slug> folder names ---
# Run `ls /kaggle/input` first to see the exact slugs Kaggle assigned.
ln -sfn /kaggle/input/mindlarge-train      data/raw/MINDlarge_train
ln -sfn /kaggle/input/mindlarge-dev        data/raw/MINDlarge_dev
ln -sfn /kaggle/input/mindlarge-test       data/raw/MINDlarge_test
ln -sfn /kaggle/input/ebnerd-large         data/raw/ebnerd_large
ln -sfn /kaggle/input/ebnerd-testset       data/raw/ebnerd_testset
# New in the latest assignment PDF: a supplementary EB-NeRD large article
# catalog. clean.py merges any article_ids from this into ebnerd_large's
# own articles.parquet (best-effort; skips cleanly if the schema surprises
# it, see src/data_pipeline/clean.py).
ln -sfn /kaggle/input/articles-large-only  data/raw/articles_large_only
# ---------------------------------------------------------------------

echo "data/raw now contains:"
ls -la data/raw

echo "======================================"
echo " Building the pipeline (clean -> split -> feature store -> embed -> BM25 -> ranker) "
echo "======================================"
PYTHONPATH=. python build_pipeline.py --dataset both

echo "======================================"
echo " Generating Codabench submissions "
echo "======================================"
PYTHONPATH=. python -m src.evaluation.generate_submission --dataset both

echo "======================================"
echo " Done. Download these from Kaggle's Output panel: "
echo "   MINDlarge_test_predictions.zip"
echo "   ebnerd_testset_predictions.zip"
echo "======================================"
