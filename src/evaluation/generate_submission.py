import os
import argparse
import zipfile
import logging
import numpy as np
import pandas as pd
import faiss
import polars as pl
from pathlib import Path

from src.retrieval.bm25 import BM25Index
from src.retrieval.features import bm25_score_candidates, build_candidate_features, compute_popularity_fast, recent_history, Ranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return Path(root) / filename
    return None

def load_test_article_catalog(test_dir, is_mind):
    """Read the test bundle's own article metadata, if shipped, so articles
    that only exist in the test period (common for EB-NeRD, whose news pool
    turns over fast) can still be embedded/indexed even when they're absent
    from the train-period feature store catalog.
    """
    if is_mind:
        news_path = find_file(test_dir, "news.tsv")
        if not news_path:
            return None
        cols = ['article_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        df = pl.read_csv(news_path, separator='\t', has_header=False, new_columns=cols,
                          quote_char=None, truncate_ragged_lines=True)
        df = df.select([
            pl.col('article_id').cast(pl.Utf8),
            pl.col('title').fill_null(''),
            pl.col('abstract').fill_null(''),
            pl.lit(None, dtype=pl.Datetime).alias('published_time'),
        ])
    else:
        art_path = find_file(test_dir, "articles.parquet")
        if not art_path:
            return None
        df = pl.read_parquet(art_path)
        subtitle_expr = pl.col('subtitle').fill_null('') if 'subtitle' in df.columns else pl.lit('')
        published_expr = pl.col('published_time') if 'published_time' in df.columns else pl.lit(None, dtype=pl.Datetime)
        df = df.select([
            pl.col('article_id').cast(pl.Utf8),
            pl.col('title').fill_null(''),
            subtitle_expr.alias('abstract'),
            published_expr.alias('published_time'),
        ])
    return df.to_pandas()

def parse_mind_time(s):
    if not s:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return None

def generate_submission(dataset_name, raw_test_dir, ranker_dataset_name=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)
    test_dir = Path(base_dir) / 'data' / 'raw' / raw_test_dir

    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}. Please ensure test zip is extracted.")
        return

    logging.info(f"--- Generating Codabench Submission for {dataset_name} ---")

    # 1. Load base article catalog + embeddings
    logging.info("Loading article catalog, embeddings, and BM25 index...")
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    id_to_idx = {str(aid): i for i, aid in enumerate(articles['article_id'])}
    embeddings = np.load(os.path.join(store_dir, 'article_embeddings.npy')).astype('float32')
    faiss.normalize_L2(embeddings)
    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    article_text = dict(zip(articles['article_id'].astype(str), articles['full_text']))
    published_col = articles['published_time'] if 'published_time' in articles.columns else pd.Series([None] * len(articles))
    id_to_published = dict(zip(articles['article_id'].astype(str), published_col))

    bm25_path = os.path.join(store_dir, 'bm25_index.pkl')
    bm25 = BM25Index.load(bm25_path) if os.path.exists(bm25_path) else None
    if bm25 is None:
        logging.warning(f"No BM25 index at {bm25_path} -- lexical score will be 0 for every candidate.")

    # 2. Detect Dataset Type and Load Behaviors
    is_mind = False
    beh_path_tsv = find_file(test_dir, "behaviors.tsv")
    user_to_history = {}

    if beh_path_tsv:
        is_mind = True
        logging.info(f"Detected MIND TSV format at {beh_path_tsv}")
        BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
        behaviors_test = pl.read_csv(
            beh_path_tsv, separator="\t", quote_char=None, has_header=False,
            new_columns=BEHAVIOR_COLS, schema_overrides={"impression_id": pl.Utf8, "user_id": pl.Utf8}
        )
    else:
        beh_path_pq = find_file(test_dir, "behaviors.parquet")
        if not beh_path_pq:
            logging.error(f"Could not find behaviors file (.tsv or .parquet) for test set in {test_dir}!")
            return

        logging.info(f"Detected EB-NeRD Parquet format at {beh_path_pq}")
        behaviors_test = pl.read_parquet(beh_path_pq).with_columns([
            pl.col("impression_id").cast(pl.Utf8),
            pl.col("user_id").cast(pl.Utf8)
        ])

        hist_path_pq = find_file(test_dir, "history.parquet")
        if hist_path_pq:
            logging.info(f"Loading Test set history from {hist_path_pq}...")
            hist_df = pl.read_parquet(hist_path_pq)
            for row in hist_df.iter_rows(named=True):
                uid = str(row['user_id'])
                h = row.get('article_id_fixed', [])
                if isinstance(h, np.ndarray):
                    h = h.tolist()
                user_to_history[uid] = [str(x) for x in h]
        else:
            logging.warning("Test history.parquet not found! Falling back to training users.parquet.")
            users_fallback = pd.read_parquet(os.path.join(store_dir, 'users.parquet'))
            user_to_history = {str(uid): list(hist) for uid, hist in zip(users_fallback['user_id'], users_fallback['history'])}

    # 3. Close the test-period article coverage gap: merge in any articles
    # the test bundle ships that the base catalog above doesn't have (the
    # bug that most likely explains a near-random EB-NeRD score -- its
    # candidates were being scored against embeddings/BM25 stats built only
    # from the ebnerd_large TRAIN window, not the test window).
    test_catalog = load_test_article_catalog(test_dir, is_mind)
    if test_catalog is not None:
        missing = test_catalog[~test_catalog['article_id'].isin(id_to_idx.keys())]
        if len(missing) > 0:
            logging.warning(f"{len(missing)} test-period articles are missing from the '{dataset_name}' "
                             f"catalog -- encoding them now so every candidate is scoreable.")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            texts = (missing['title'].fillna('') + " " + missing['abstract'].fillna('')).tolist()
            new_embeds = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(new_embeds)
            embeddings = np.vstack([embeddings, new_embeds])
            start_idx = len(id_to_idx)
            for offset, (aid, text, published) in enumerate(zip(missing['article_id'], texts, missing['published_time'])):
                aid = str(aid)
                id_to_idx[aid] = start_idx + offset
                article_text[aid] = text
                id_to_published[aid] = published
            if bm25 is not None:
                added = bm25.add_documents(dict(zip(missing['article_id'].astype(str), texts)))
                logging.info(f"Incorporated {added} missing articles into the in-memory BM25 index.")
        else:
            logging.info("Test-period article catalog is fully covered by the existing feature store.")

    # 4. Load the hybrid ranker (trained on the labelled train split) if available
    ranker_store = os.path.join(base_dir, 'data', 'feature_store', ranker_dataset_name) if ranker_dataset_name else store_dir
    ranker_path = os.path.join(ranker_store, 'ranker.pkl')
    ranker = Ranker.load(ranker_path) if os.path.exists(ranker_path) else None
    if ranker:
        logging.info(f"Loaded hybrid ranker from {ranker_path}.")
        popularity = ranker.popularity
    else:
        logging.warning("No trained ranker found -- falling back to semantic-only scoring "
                         "with a popularity fallback for cold-start users.")
        # Prefer the validation window (closest in time to test) over the
        # full train history -- see the comment in train_ranker.py: a
        # stale, all-of-train popularity prior can be actively
        # anti-predictive once the news cycle has moved on.
        val_path = os.path.join(ranker_store, 'impressions_val.parquet')
        train_path = os.path.join(ranker_store, 'impressions_train.parquet')
        pop_path = val_path if os.path.exists(val_path) else train_path
        popularity = compute_popularity_fast(pop_path) if os.path.exists(pop_path) else {}

    total_rows = behaviors_test.height
    logging.info(f"Found {total_rows:,} test impressions.")

    output_file = os.path.join(base_dir, f"{raw_test_dir}_predictions.txt")

    # Resume support: at 13.5M/2.37M test impressions, a single run can
    # comfortably exceed a Kaggle session's time limit. If a previous
    # attempt got partway through and was killed, don't redo that work --
    # pick up where it left off instead of silently overwriting it.
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'rb+') as f:
            content = f.read()
            if content and not content.endswith(b'\n'):
                # The process was killed mid-write of the last line -- drop
                # that possibly-truncated line rather than trust it.
                last_nl = content.rfind(b'\n')
                content = content[:last_nl + 1]
                f.seek(last_nl + 1)
                f.truncate()
            for line in content.splitlines():
                if line:
                    done_ids.add(line.split(b' ', 1)[0].decode())
        if done_ids:
            logging.info(f"Resuming: found {len(done_ids):,} predictions already written in "
                         f"{output_file} -- skipping those instead of redoing them.")

    with open(output_file, 'a' if done_ids else 'w') as f:
        for count, row in enumerate(behaviors_test.iter_rows(named=True)):
            if count > 0 and count % 50000 == 0:
                logging.info(f"  Processed {count:,}/{total_rows:,} predictions...")
                f.flush()
                os.fsync(f.fileno())

            imp_id = str(row['impression_id'])
            if imp_id in done_ids:
                continue
            user_id = str(row['user_id'])

            if is_mind:
                history_str = row['history']
                history = history_str.split(" ") if history_str else []
                imp_str = row['impressions']
                candidates = imp_str.split(" ") if imp_str else []
                ref_time = parse_mind_time(row.get('time'))
            else:
                history = user_to_history.get(user_id, [])
                candidates = row.get('article_ids_inview', [])
                if isinstance(candidates, np.ndarray):
                    candidates = candidates.tolist()
                candidates = [str(c) for c in candidates]
                ref_time = row.get('impression_time')

            if not candidates:
                f.write(f"{imp_id} []\n")
                continue

            hist_len = len(history)
            recent = recent_history(history)
            history_indices = [id_to_idx[aid] for aid in recent if aid in id_to_idx]

            if not history_indices:
                user_vector = np.zeros((1, embeddings.shape[1]), dtype='float32')
            else:
                user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
                faiss.normalize_L2(user_vector)

            # Vectorized semantic scoring (matrix multiply instead of a
            # per-candidate Python loop -- matters once this runs over
            # MINDlarge/ebnerd_large scale test sets).
            candidate_indices = np.array([id_to_idx.get(aid, -1) for aid in candidates])
            valid_mask = candidate_indices >= 0
            sem_scores = np.zeros(len(candidates), dtype=np.float64)
            if valid_mask.any():
                cand_vecs = embeddings[candidate_indices[valid_mask]]
                sem_scores[valid_mask] = np.dot(user_vector, cand_vecs.T)[0]

            if bm25 is not None:
                query_text = " ".join(article_text.get(aid, "") for aid in recent)
                query_tokens = set(bm25.tokenize(query_text))
                lex_scores = bm25_score_candidates(bm25, query_tokens, candidates)
            else:
                lex_scores = np.zeros(len(candidates), dtype=np.float64)

            if ranker:
                X = build_candidate_features(candidates, lex_scores, sem_scores, popularity,
                                              id_to_published, ref_time, hist_len)
                final_scores = ranker.predict(X)
            elif not history_indices:
                # No ranker to lean on for cold-start users: popularity is a
                # far better prior than the previous all-zero-score tie
                # (which ranked candidates in arbitrary input order).
                final_scores = np.array([popularity.get(aid, 0.0) for aid in candidates])
            else:
                final_scores = sem_scores

            # Convert scores to dense ranks (1 is highest score)
            order = np.argsort(final_scores)[::-1]
            ranks = np.zeros(len(final_scores), dtype=int)
            ranks[order] = np.arange(1, len(final_scores) + 1)

            f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")

    # 5. Zip it up for Codabench
    zip_file = os.path.join(base_dir, f"{raw_test_dir}_predictions.zip")
    logging.info(f"Zipping {output_file} into {zip_file}...")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        arcname = "prediction.txt" if is_mind else "predictions.txt"
        zf.write(output_file, arcname=arcname)

    logging.info(f"Success! Ready for Codabench upload: {zip_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Codabench submission files.")
    parser.add_argument('--dataset', type=str, choices=['mind', 'ebnerd', 'both'], default='both',
                        help='Which dataset to process (mind, ebnerd, or both)')
    args = parser.parse_args()

    if args.dataset in ['mind', 'both']:
        generate_submission('MINDlarge_test', 'MINDlarge_test', ranker_dataset_name='MINDlarge_train')

    if args.dataset in ['ebnerd', 'both']:
        generate_submission('ebnerd_large', 'ebnerd_testset', ranker_dataset_name='ebnerd_large')
