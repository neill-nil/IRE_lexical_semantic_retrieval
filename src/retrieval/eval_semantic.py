import os
import json
import logging
import pandas as pd
import numpy as np
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_semantic(dataset_name):
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
    
    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    user_hist_dict = dict(zip(users['user_id'], users['history']))
    
    impressions = pd.read_parquet(os.path.join(store_dir, 'impressions_test.parquet'))
    
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
        unclicked = row['unclicked_articles']
        
        if clicked is None or len(clicked) == 0:
            continue
            
        ground_truth = set(clicked)
        inview_candidates = list(set(clicked) | set(unclicked))
        
        # 3A. Build User Representation (Mean-Pooling)
        history = user_hist_dict.get(user_id, [])
        history_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
        
        if not history_indices:
            # Cold-start user: default to zero vector
            user_vector = np.zeros((1, dim), dtype='float32')
        else:
            # Mean pooling
            user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
            # Re-normalize user vector for pure cosine similarity
            faiss.normalize_L2(user_vector)
            
        # 3B. Score Candidate Articles
        candidate_indices = [id_to_idx[aid] for aid in inview_candidates if aid in id_to_idx]
        
        # Fast Dot Product scoring between User Vector and Candidate Vectors
        if not candidate_indices:
            continue
            
        candidate_vectors = embeddings[candidate_indices]
        # user_vector is (1, 384), candidate_vectors is (C, 384)
        # Dot product yields (1, C)
        scores = np.dot(user_vector, candidate_vectors.T)[0]
        
        # 3C. Rank and calculate Recall
        # Sort candidates by score descending
        ranked_indices = np.argsort(scores)[::-1]
        ranked_candidates = [inview_candidates[i] for i in ranked_indices]
        
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
    datasets = ['MINDsmall_train', 'ebnerd_small']
    for ds in datasets:
        evaluate_semantic(ds)
