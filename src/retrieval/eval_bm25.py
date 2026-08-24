import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from src.retrieval.bm25 import BM25Index
from src.retrieval.features import bm25_retrieve_topk, recent_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_dataset(dataset_name, sample_size=None, seed=42):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)

    index_path = os.path.join(store_dir, 'bm25_index.pkl')
    if not os.path.exists(index_path):
        logging.error(f"No BM25 index found for {dataset_name}. Run bm25.py first.")
        return

    logging.info(f"--- Evaluating {dataset_name} ---")
    bm25 = BM25Index.load(index_path)

    # Load feature store
    logging.info("Loading feature store catalogs...")
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    article_text_dict = dict(zip(articles['article_id'], articles['full_text']))

    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    user_hist_dict = dict(zip(users['user_id'], users['history']))

    test_path = os.path.join(store_dir, 'impressions_test.parquet')
    if not os.path.exists(test_path):
        logging.warning(f"No test set found for {dataset_name}.")
        return
    impressions = pd.read_parquet(test_path)
    if sample_size and len(impressions) > sample_size:
        logging.info(f"Sampling {sample_size:,}/{len(impressions):,} test impressions for full-corpus retrieval...")
        impressions = impressions.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    
    # Metrics
    recalls = {50: [], 100: [], 200: []}
    
    total_impressions = len(impressions)
    logging.info(f"Scoring {total_impressions} impressions in the test set...")

    for count, (idx, row) in enumerate(impressions.iterrows()):
        if count > 0 and count % 500 == 0:
            logging.info(f"  Processed {count}/{total_impressions} impressions...")

        user_id = row['user_id']
        clicked = row['clicked_articles']

        # In rare cases, clicked might be None or empty
        if clicked is None or len(clicked) == 0:
            continue
            
        # Ground truth: the clicked article(s) must be retrievable out of the
        # FULL article corpus, not just the handful of items the impression
        # already narrowed down to -- that's what makes this a retrieval
        # metric rather than a reranking metric.
        ground_truth = set(clicked)

        # Build pseudo-query from the user's most recent history
        history = recent_history(user_hist_dict.get(user_id, []))
        query_text = " ".join([article_text_dict.get(aid, "") for aid in history])
        query_tokens = set(bm25.tokenize(query_text))

        # Full-corpus BM25 retrieval (top-200 covers every K we report)
        ranked_candidates = bm25_retrieve_topk(bm25, query_tokens, k=200)

        # Evaluate Recall@K
        for K in [50, 100, 200]:
            top_k = set(ranked_candidates[:K])
            # Recall = (number of correctly predicted clicks) / (total actual clicks)
            hits = len(ground_truth.intersection(top_k))
            recalls[K].append(hits / len(ground_truth))
            
    # Final Metric Aggregation
    logging.info(f"Results for {dataset_name}:")
    
    saved_metrics = {}
    for K in [50, 100, 200]:
        mean_recall = np.mean(recalls[K]) if recalls[K] else 0.0
        saved_metrics[f"Recall@{K}"] = float(mean_recall)
        logging.info(f"  Recall@{K}: {mean_recall:.4f}")
        
    metrics_path = os.path.join(store_dir, 'metrics_bm25.json')
    with open(metrics_path, 'w') as f:
        json.dump(saved_metrics, f, indent=4)
    logging.info(f"Saved BM25 metrics to {metrics_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval recall@K.")
    parser.add_argument('--dataset', type=str, default=None,
                        help='A single feature_store dataset name to evaluate (default: all with a bm25_index.pkl)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Cap the number of test impressions scored (recommended for MINDlarge/ebnerd_large)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_root = os.path.join(base_dir, 'data', 'feature_store')
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [d for d in os.listdir(store_root)
                    if os.path.exists(os.path.join(store_root, d, 'bm25_index.pkl'))]
    for ds in datasets:
        evaluate_dataset(ds, sample_size=args.sample_size)
