import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from src.retrieval.bm25 import BM25Index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_dataset(dataset_name):
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
    
    # Metrics
    recalls = {50: [], 100: [], 200: []}
    
    total_impressions = len(impressions)
    logging.info(f"Scoring {total_impressions} impressions in the test set...")
    
    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    
    for count, (idx, row) in enumerate(impressions.iterrows()):
        if count > 0 and count % 500 == 0:
            logging.info(f"  Processed {count}/{total_impressions} impressions...")
            
        user_id = row['user_id']
        clicked = row['clicked_articles']
        unclicked = row['unclicked_articles']
        
        # In rare cases, clicked might be None or empty
        if clicked is None or len(clicked) == 0:
            continue
            
        # Ground truth and candidate pool
        ground_truth = set(clicked)
        inview_candidates = set(clicked) | set(unclicked)
        
        # Build pseudo-query from history
        history = user_hist_dict.get(user_id, [])
        query_text = " ".join([article_text_dict.get(aid, "") for aid in history])
        query_tokens = set(bm25.tokenize(query_text))
        
        # BM25 Scoring (Optimized Inverted Index Lookup)
        scores = {doc_id: 0.0 for doc_id in inview_candidates}
        
        for token in query_tokens:
            if token in bm25.inverted_index:
                idf = bm25.idf[token]
                # Look at all documents that contain this token
                for doc_id, tf in bm25.inverted_index[token].items():
                    if doc_id in inview_candidates:
                        dl = bm25.doc_lens.get(doc_id, avgdl)
                        # BM25 Formula
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
                        scores[doc_id] += idf * (numerator / denominator)
                        
        # Rank candidates by score (descending)
        ranked_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
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
    # You can specify which datasets to run evaluation on
    datasets = ['MINDsmall_train', 'ebnerd_small']
    for ds in datasets:
        evaluate_dataset(ds)
