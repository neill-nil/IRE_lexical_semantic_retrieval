"""Train a hybrid ranker (Q2 BM25 score + Q3 semantic score + popularity +
recency) that combines lexical and semantic candidate generation into a
single click-probability used for the actual Codabench submission.

The unsupervised "just use cosine similarity" baseline in
generate_submission.py has no way to learn how much to trust each signal;
a small logistic regression fit on the labelled train impressions does,
and is cheap enough to fit on a bounded sample even for MINDlarge/
ebnerd_large.
"""
import os
import logging
import argparse
import numpy as np
import pandas as pd
import faiss

from src.retrieval.bm25 import BM25Index
from src.retrieval.features import bm25_score_candidates, build_candidate_features, compute_popularity_fast, recent_history, Ranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _extract_training_rows(impressions, user_hist_dict, article_text_dict, id_to_idx,
                            embeddings, bm25, popularity, id_to_published):
    X_rows, y_rows = [], []
    total = len(impressions)
    for count, row in enumerate(impressions.itertuples(index=False)):
        if count > 0 and count % 20000 == 0:
            logging.info(f"  Extracted features for {count:,}/{total:,} sampled impressions...")

        clicked = row.clicked_articles
        unclicked = row.unclicked_articles
        if clicked is None or unclicked is None or len(clicked) == 0 or len(unclicked) == 0:
            continue

        candidates = list(clicked) + list(unclicked)
        if any(aid not in id_to_idx for aid in candidates):
            continue  # keep training data aligned; drop the rare malformed row

        history = user_hist_dict.get(row.user_id, [])
        if history is None:
            history = []
        hist_len = len(history)
        recent = recent_history(history)

        query_text = " ".join([article_text_dict.get(aid, "") for aid in recent])
        query_tokens = set(bm25.tokenize(query_text))
        lex_scores = bm25_score_candidates(bm25, query_tokens, candidates)

        history_indices = [id_to_idx[aid] for aid in recent if aid in id_to_idx]
        if not history_indices:
            user_vector = np.zeros((1, embeddings.shape[1]), dtype='float32')
        else:
            user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
            faiss.normalize_L2(user_vector)
        candidate_vectors = embeddings[[id_to_idx[aid] for aid in candidates]]
        sem_scores = np.dot(user_vector, candidate_vectors.T)[0]

        X = build_candidate_features(candidates, lex_scores, sem_scores, popularity,
                                      id_to_published, row.time, hist_len)
        y = np.array([1] * len(clicked) + [0] * len(unclicked))

        X_rows.append(X)
        y_rows.append(y)

    if not X_rows:
        return np.empty((0, 5)), np.empty((0,))
    return np.vstack(X_rows), np.concatenate(y_rows)


def train_ranker(dataset_name, sample_size=200_000, seed=42):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)

    train_path = os.path.join(store_dir, 'impressions_train.parquet')
    bm25_path = os.path.join(store_dir, 'bm25_index.pkl')
    emb_path = os.path.join(store_dir, 'article_embeddings.npy')
    if not (os.path.exists(train_path) and os.path.exists(bm25_path) and os.path.exists(emb_path)):
        logging.warning(f"Skipping ranker for {dataset_name}: missing train impressions, BM25 index, or embeddings.")
        return

    logging.info(f"--- Training Hybrid Ranker for {dataset_name} ---")

    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    id_to_idx = {aid: i for i, aid in enumerate(articles['article_id'])}
    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    article_text_dict = dict(zip(articles['article_id'], articles['full_text']))
    published_col = articles['published_time'] if 'published_time' in articles.columns else pd.Series([None] * len(articles))
    id_to_published = dict(zip(articles['article_id'], published_col))
    has_recency = published_col.notna().any()

    users = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
    user_hist_dict = dict(zip(users['user_id'], users['history']))

    logging.info("Loading BM25 index and article embeddings...")
    bm25 = BM25Index.load(bm25_path)
    embeddings = np.load(emb_path).astype('float32')
    faiss.normalize_L2(embeddings)

    # Popularity is deliberately computed from the VALIDATION window (the
    # 24h immediately preceding test, per split.py) rather than the full
    # train history. News popularity decays fast (EB-NeRD especially): an
    # article that dominated clicks weeks into the train window is often
    # stale by test time, so an all-of-train click-count prior measured
    # negatively correlated with test clicks in practice (AUC ~0.40 as a
    # standalone feature) -- a recent-window prior tracks what's *currently*
    # trending instead of what was popular historically.
    val_path = os.path.join(store_dir, 'impressions_val.parquet')
    popularity_source = val_path if os.path.exists(val_path) else train_path
    logging.info(f"Computing article popularity from {os.path.basename(popularity_source)} (recent window preferred)...")
    popularity = compute_popularity_fast(popularity_source)

    train_full = pd.read_parquet(train_path)
    if len(train_full) > sample_size:
        logging.info(f"Sampling {sample_size:,}/{len(train_full):,} train impressions for ranker fitting...")
        train_sample = train_full.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        train_sample = train_full

    X, y = _extract_training_rows(train_sample, user_hist_dict, article_text_dict, id_to_idx,
                                   embeddings, bm25, popularity, id_to_published)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        logging.warning(f"Not enough labelled data to train a ranker for {dataset_name}. Skipping.")
        return

    logging.info(f"Fitting logistic regression ranker on {len(y):,} candidate rows ({y.mean():.4f} positive rate)...")
    ranker = Ranker(popularity=popularity, has_recency=bool(has_recency)).fit(X, y)

    # Quick sanity check: does the hybrid actually beat its own inputs?
    from sklearn.metrics import roc_auc_score
    preds = ranker.predict(X)
    logging.info(f"  Hybrid train AUC:   {roc_auc_score(y, preds):.4f}")
    logging.info(f"  BM25-only train AUC:     {roc_auc_score(y, X[:, 0]):.4f}")
    logging.info(f"  Semantic-only train AUC: {roc_auc_score(y, X[:, 1]):.4f}")

    out_path = os.path.join(store_dir, 'ranker.pkl')
    ranker.save(out_path)
    logging.info(f"Saved hybrid ranker to {out_path}")


def run(dataset='both'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_root = os.path.join(base_dir, 'data', 'feature_store')
    if not os.path.isdir(store_root):
        return
    all_dirs = [d for d in os.listdir(store_root) if os.path.isdir(os.path.join(store_root, d))]
    if dataset == 'mind':
        datasets = [d for d in all_dirs if 'MIND' in d]
    elif dataset == 'ebnerd':
        datasets = [d for d in all_dirs if 'ebnerd' in d]
    else:
        datasets = all_dirs

    for ds in datasets:
        train_ranker(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the hybrid ranking model.")
    parser.add_argument('--dataset', type=str, choices=['mind', 'ebnerd', 'both'], default='both')
    parser.add_argument('--sample-size', type=int, default=200_000,
                        help='Max train impressions to extract features from (fitting itself is fast; feature extraction is the bottleneck).')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_root = os.path.join(base_dir, 'data', 'feature_store')
    all_dirs = [d for d in os.listdir(store_root) if os.path.isdir(os.path.join(store_root, d))] if os.path.isdir(store_root) else []
    if args.dataset == 'mind':
        datasets = [d for d in all_dirs if 'MIND' in d]
    elif args.dataset == 'ebnerd':
        datasets = [d for d in all_dirs if 'ebnerd' in d]
    else:
        datasets = all_dirs
    for ds in datasets:
        train_ranker(ds, sample_size=args.sample_size)
