import os
import json
import logging
import pandas as pd
import numpy as np
import faiss
from sklearn.metrics import roc_auc_score, ndcg_score
from src.retrieval.bm25 import BM25Index
from src.retrieval.features import bm25_score_candidates, build_candidate_features, recent_history, Ranker

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

def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42, max_n=20000):
    """Bootstrap 95% CI (default) over a metric's per-impression values.
    Caps the resampled population at `max_n` so this stays a vectorized,
    O(n_boot * max_n) computation even when the underlying test set has
    millions of impressions (MINDlarge/ebnerd_large) -- a full-data
    bootstrap at that scale would blow both time and memory budgets.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    if len(values) > max_n:
        values = rng.choice(values, size=max_n, replace=False)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)

def evaluate_harness(dataset_name, sample_size=None, seed=42):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)

    logging.info(f"--- Running Q4 Harness on {dataset_name} ---")

    # 1. Load Data
    logging.info("Loading Articles, Users, and Test Impressions...")
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    impressions = pd.read_parquet(os.path.join(store_dir, 'impressions_test.parquet'))
    if sample_size and len(impressions) > sample_size:
        logging.info(f"Sampling {sample_size:,}/{len(impressions):,} test impressions for evaluation...")
        impressions = impressions.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    id_to_idx = {aid: i for i, aid in enumerate(articles['article_id'])}
    id_to_category = dict(zip(articles['article_id'], articles['category']))
    user_hist_dict = dict(zip(users['user_id'], users['history']))

    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    article_text_dict = dict(zip(articles['article_id'], articles['full_text']))
    id_to_published = dict(zip(articles['article_id'], articles.get('published_time', pd.Series(dtype='datetime64[ns]'))))

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

    # 3b. Optional Hybrid Ranker (trained by train_ranker.py). Adding this
    # as a third method lets the design note actually demonstrate whether
    # combining lexical + semantic + popularity + recency beats either
    # signal alone, instead of asserting it.
    ranker_path = os.path.join(store_dir, 'ranker.pkl')
    ranker = Ranker.load(ranker_path) if os.path.exists(ranker_path) else None
    if ranker:
        logging.info("Loaded Hybrid ranker -- will also score a 'Hybrid' method.")
    methods = ['Lexical', 'Semantic'] + (['Hybrid'] if ranker else [])

    # 4. Evaluation Loop

    metrics = {m: [] for m in methods}

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

        # NOTE: users with zero click history are intentionally kept (not
        # skipped) so the cold-start slice below reflects real users
        # instead of being silently emptied out.
        history = user_hist_dict.get(user_id, [])
        if history is None:
            history = []
        hist_len = len(history)  # full history length, used for cold/warm slicing
        recent = recent_history(history)  # bounded window actually used for scoring

        # ---------------------------
        # LEXICAL SCORING (BM25)
        # ---------------------------
        query_text = " ".join([article_text_dict.get(aid, "") for aid in recent])
        query_tokens = set(bm25.tokenize(query_text))
        lex_scores_arr = bm25_score_candidates(bm25, query_tokens, candidates)

        # ---------------------------
        # SEMANTIC SCORING (FAISS)
        # ---------------------------
        history_indices = [id_to_idx[aid] for aid in recent if aid in id_to_idx]
        if not history_indices:
            # True cold-start: no informative signal, matching the
            # cold-start handling used at submission time.
            user_vector = np.zeros((1, dim), dtype='float32')
        else:
            user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
            faiss.normalize_L2(user_vector)

        candidate_indices = [id_to_idx[aid] for aid in candidates if aid in id_to_idx]
        if len(candidate_indices) != len(candidates):
            continue # Skip malformed candidates
        candidate_vectors = embeddings[candidate_indices]
        sem_scores_arr = np.dot(user_vector, candidate_vectors.T)[0]

        # ---------------------------
        # CALCULATE METRICS
        # ---------------------------
        y_true_arr = np.array(y_true)

        method_scores = {'Lexical': lex_scores_arr, 'Semantic': sem_scores_arr}
        if ranker:
            ref_time = row.get('time', None)
            X = build_candidate_features(
                candidates, lex_scores_arr, sem_scores_arr, ranker.popularity,
                id_to_published, ref_time, hist_len
            )
            method_scores['Hybrid'] = ranker.predict(X)

        for method, scores_arr in method_scores.items():
            top_5 = np.array(candidates)[np.argsort(scores_arr)[::-1][:5]]
            metrics[method].append({
                'hist_len': hist_len,
                'AUC': roc_auc_score(y_true_arr, scores_arr),
                'MRR': mrr_score(y_true_arr, scores_arr),
                'nDCG@5': ndcg_score([y_true_arr], [scores_arr], k=5),
                'nDCG@10': ndcg_score([y_true_arr], [scores_arr], k=10),
                'ILD@5': ild_score(top_5, id_to_category)
            })

    # Aggregate and Save (Q5 Slicing + bootstrap CIs)
    final_metrics = {}
    for model in methods:
        final_metrics[model] = {'Global': {}, 'Cold (History < 5)': {}, 'Warm (History >= 5)': {}}
        logging.info(f"--- {model} Results for {dataset_name} ---")

        df = pd.DataFrame(metrics[model])
        if df.empty:
            continue

        cold_df = df[df['hist_len'] < 5]
        warm_df = df[df['hist_len'] >= 5]

        for m in ['AUC', 'MRR', 'nDCG@5', 'nDCG@10', 'ILD@5']:
            for slice_name, slice_df in [('Global', df), ('Cold (History < 5)', cold_df), ('Warm (History >= 5)', warm_df)]:
                if slice_df.empty:
                    final_metrics[model][slice_name][m] = {'mean': 0.0, 'ci95': [0.0, 0.0], 'n': 0}
                    continue
                mean_val = float(slice_df[m].mean())
                lo, hi = bootstrap_ci(slice_df[m].values)
                final_metrics[model][slice_name][m] = {'mean': mean_val, 'ci95': [lo, hi], 'n': int(len(slice_df))}

            g = final_metrics[model]['Global'][m]['mean']
            c = final_metrics[model]['Cold (History < 5)'][m]['mean']
            w = final_metrics[model]['Warm (History >= 5)'][m]['mean']
            logging.info(f"  {m}: Global={g:.4f} | Cold={c:.4f} (n={final_metrics[model]['Cold (History < 5)'][m]['n']}) | Warm={w:.4f} (n={final_metrics[model]['Warm (History >= 5)'][m]['n']})")

    out_path = os.path.join(store_dir, 'metrics_q4_q5.json')
    with open(out_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the Q4/Q5 offline evaluation harness.")
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
        evaluate_harness(ds, sample_size=args.sample_size)
