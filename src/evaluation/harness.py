import os
import json
import logging
import pandas as pd
import numpy as np
import faiss
import pickle
import math
from sklearn.metrics import roc_auc_score, ndcg_score
from src.retrieval.bm25 import BM25Index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mrr_score(y_true, y_score):
    # Sort descending by score
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    
    # Find the index of the first '1'
    ones = np.where(y_true_sorted == 1)[0]
    if len(ones) == 0:
        return 0.0
    return 1.0 / (ones[0] + 1)

def ild_score(top_k_indices, id_to_category):
    if len(top_k_indices) == 0: return 0.0
    categories = [id_to_category.get(idx, "unknown") for idx in top_k_indices]
    unique_categories = len(set(categories))
    # Diversity = Ratio of unique categories in the Top-K (1.0 = perfectly diverse)
    return unique_categories / len(top_k_indices)

def evaluate_harness(dataset_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)
    
    logging.info(f"--- Running Q4 Harness on {dataset_name} ---")
    
    # 1. Load Data
    logging.info("Loading Articles, Users, and Test Impressions...")
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    impressions = pd.read_parquet(os.path.join(store_dir, 'impressions_test.parquet'))
    
    id_to_idx = {aid: i for i, aid in enumerate(articles['article_id'])}
    id_to_category = dict(zip(articles['article_id'], articles['category']))
    user_hist_dict = dict(zip(users['user_id'], users['history']))
    
    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    article_text_dict = dict(zip(articles['article_id'], articles['full_text']))
    
    # 2. Setup Lexical (BM25)
    logging.info("Loading BM25 Index...")
    bm25_path = os.path.join(store_dir, 'bm25_index.pkl')
    bm25 = BM25Index.load(bm25_path)
        
    # 3. Setup Semantic (FAISS)
    logging.info("Loading FAISS Index...")
    embeddings = np.load(os.path.join(store_dir, 'article_embeddings.npy')).astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 4. Evaluation Loop
        
    metrics = {
        'Lexical': [],
        'Semantic': []
    }
    
    total = len(impressions)
    for count, (row_idx, row) in enumerate(impressions.iterrows()):
        if count > 0 and count % 500 == 0:
            logging.info(f"Processed {count}/{total} impressions...")
            
        user_id = row['user_id']
        clicked = row['clicked_articles']
        unclicked = row['unclicked_articles']
        
        if clicked is None or unclicked is None or len(clicked) == 0 or len(unclicked) == 0:
            continue
            
        candidates = list(clicked) + list(unclicked)
        y_true = [1]*len(clicked) + [0]*len(unclicked)
        
        history = user_hist_dict.get(user_id, [])
        if history is None or len(history) == 0:
            continue
            
        hist_len = len(history)
            
        # ---------------------------
        # LEXICAL SCORING (BM25)
        # ---------------------------
        query_text = " ".join([article_text_dict.get(aid, "") for aid in history])
        query_tokens = set(bm25.tokenize(query_text))
        
        lex_scores = []
        for aid in candidates:
            score = 0.0
            dl = bm25.doc_lens.get(aid, bm25.avgdl)
            for token in query_tokens:
                if token in bm25.inverted_index and aid in bm25.inverted_index[token]:
                    tf = bm25.inverted_index[token][aid]
                    idf = bm25.idf.get(token, 0)
                    numerator = tf * (bm25.k1 + 1)
                    denominator = tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avgdl)
                    score += idf * (numerator / denominator)
            lex_scores.append(score)
            
        # ---------------------------
        # SEMANTIC SCORING (FAISS)
        # ---------------------------
        history_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
        user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
        faiss.normalize_L2(user_vector)
        
        candidate_indices = [id_to_idx[aid] for aid in candidates if aid in id_to_idx]
        if len(candidate_indices) != len(candidates):
            continue # Skip malformed candidates
        candidate_vectors = embeddings[candidate_indices]
        sem_scores = np.dot(user_vector, candidate_vectors.T)[0]
        
        # ---------------------------
        # CALCULATE METRICS
        # ---------------------------
        y_true_arr = np.array(y_true)
        lex_scores_arr = np.array(lex_scores)
        sem_scores_arr = np.array(sem_scores)
        
        lex_top_5 = np.array(candidates)[np.argsort(lex_scores_arr)[::-1][:5]]
        sem_top_5 = np.array(candidates)[np.argsort(sem_scores_arr)[::-1][:5]]
        
        lex_metrics = {
            'hist_len': hist_len,
            'AUC': roc_auc_score(y_true_arr, lex_scores_arr),
            'MRR': mrr_score(y_true_arr, lex_scores_arr),
            'nDCG@5': ndcg_score([y_true_arr], [lex_scores_arr], k=5),
            'nDCG@10': ndcg_score([y_true_arr], [lex_scores_arr], k=10),
            'ILD@5': ild_score(lex_top_5, id_to_category)
        }
        
        sem_metrics = {
            'hist_len': hist_len,
            'AUC': roc_auc_score(y_true_arr, sem_scores_arr),
            'MRR': mrr_score(y_true_arr, sem_scores_arr),
            'nDCG@5': ndcg_score([y_true_arr], [sem_scores_arr], k=5),
            'nDCG@10': ndcg_score([y_true_arr], [sem_scores_arr], k=10),
            'ILD@5': ild_score(sem_top_5, id_to_category)
        }
        
        metrics['Lexical'].append(lex_metrics)
        metrics['Semantic'].append(sem_metrics)
        
    # Aggregate and Save (Q5 Slicing)
    final_metrics = {}
    for model in ['Lexical', 'Semantic']:
        final_metrics[model] = {'Global': {}, 'Cold (History < 5)': {}, 'Warm (History >= 5)': {}}
        logging.info(f"--- {model} Results for {dataset_name} ---")
        
        df = pd.DataFrame(metrics[model])
        if df.empty:
            continue
            
        cold_df = df[df['hist_len'] < 5]
        warm_df = df[df['hist_len'] >= 5]
        
        for m in ['AUC', 'MRR', 'nDCG@5', 'nDCG@10', 'ILD@5']:
            val_global = float(df[m].mean())
            val_cold = float(cold_df[m].mean()) if not cold_df.empty else 0.0
            val_warm = float(warm_df[m].mean()) if not warm_df.empty else 0.0
            
            final_metrics[model]['Global'][m] = val_global
            final_metrics[model]['Cold (History < 5)'][m] = val_cold
            final_metrics[model]['Warm (History >= 5)'][m] = val_warm
            
            logging.info(f"  {m}: Global={val_global:.4f} | Cold={val_cold:.4f} | Warm={val_warm:.4f}")
            
    out_path = os.path.join(store_dir, 'metrics_q4_q5.json')
    with open(out_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
if __name__ == '__main__':
    for ds in ['MINDsmall_train', 'ebnerd_small']:
        evaluate_harness(ds)
