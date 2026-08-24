import os
import json
import logging
import pandas as pd
import numpy as np
import faiss

from src.retrieval.features import recent_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_semantic(dataset_name, sample_size=None, seed=42):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)
    
    embeddings_path = os.path.join(store_dir, 'article_embeddings.npy')
    if not os.path.exists(embeddings_path):
        logging.error(f"No embeddings found for {dataset_name}. Run embed.py first.")
        return
        
    logging.info(f"--- Evaluating Semantic Retrieval on {dataset_name} ---")
    
    # 1. Load Data
    logging.info("Loading vectors and catalogs...")
    embeddings = np.load(embeddings_path).astype('float32')
    
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    # Create mapping from article_id to vector matrix index
    id_to_idx = {aid: i for i, aid in enumerate(articles['article_id'])}
    idx_to_id = {i: aid for aid, i in id_to_idx.items()}
    
    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    user_hist_dict = dict(zip(users['user_id'], users['history']))
    
    impressions = pd.read_parquet(os.path.join(store_dir, 'impressions_test.parquet'))
    if sample_size and len(impressions) > sample_size:
        logging.info(f"Sampling {sample_size:,}/{len(impressions):,} test impressions for evaluation...")
        impressions = impressions.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # 2. Build FAISS Index
    # We L2 normalize the vectors so that Dot Product == Cosine Similarity
    logging.info("Normalizing vectors and building FAISS IndexFlatIP...")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 3. Evaluation Loop
    total_impressions = len(impressions)
    logging.info(f"Scoring {total_impressions} impressions in the test set...")
    
    recalls = {50: [], 100: [], 200: []}
    
    for count, (row_idx, row) in enumerate(impressions.iterrows()):
        if count > 0 and count % 500 == 0:
            logging.info(f"  Processed {count}/{total_impressions} impressions...")
            
        user_id = row['user_id']
        clicked = row['clicked_articles']

        if clicked is None or len(clicked) == 0:
            continue

        # Ground truth: recall is measured against the FULL article corpus
        # (via the FAISS index built over all articles), not just the
        # handful of items already surfaced in this impression -- otherwise
        # Recall@50 is trivially ~1.0 whenever the impression has fewer
        # than 50 candidates to begin with.
        ground_truth = set(clicked)

        # 3A. Build User Representation (Mean-Pooling over recent history)
        history = recent_history(user_hist_dict.get(user_id, []))
        history_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]

        if not history_indices:
            # Cold-start user: default to zero vector
            user_vector = np.zeros((1, dim), dtype='float32')
        else:
            # Mean pooling
            user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
            # Re-normalize user vector for pure cosine similarity
            faiss.normalize_L2(user_vector)

        # 3B. True ANN retrieval: search the full FAISS index for the
        # user's top-200 nearest articles by cosine similarity.
        _, top_indices = index.search(user_vector, 200)
        ranked_candidates = [idx_to_id[i] for i in top_indices[0] if i != -1]

        for K in [50, 100, 200]:
            top_k = set(ranked_candidates[:K])
            hits = len(ground_truth.intersection(top_k))
            recalls[K].append(hits / len(ground_truth))
            
    # Final Metric Aggregation
    logging.info(f"Semantic Results for {dataset_name}:")
    
    saved_metrics = {}
    for K in [50, 100, 200]:
        mean_recall = np.mean(recalls[K]) if recalls[K] else 0.0
        saved_metrics[f"Recall@{K}"] = float(mean_recall)
        logging.info(f"  Recall@{K}: {mean_recall:.4f}")
        
    metrics_path = os.path.join(store_dir, 'metrics_semantic.json')
    with open(metrics_path, 'w') as f:
        json.dump(saved_metrics, f, indent=4)
    logging.info(f"Saved Semantic metrics to {metrics_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval recall@K.")
    parser.add_argument('--dataset', type=str, default=None,
                        help='A single feature_store dataset name to evaluate (default: all with article_embeddings.npy)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Cap the number of test impressions scored (recommended for MINDlarge/ebnerd_large)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_root = os.path.join(base_dir, 'data', 'feature_store')
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [d for d in os.listdir(store_root)
                    if os.path.exists(os.path.join(store_root, d, 'article_embeddings.npy'))]
    for ds in datasets:
        evaluate_semantic(ds, sample_size=args.sample_size)
