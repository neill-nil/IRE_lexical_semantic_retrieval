import os
import zipfile
import logging
import numpy as np
import pandas as pd
import faiss
import polars as pl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return Path(root) / filename
    return None

def generate_submission(dataset_name, raw_test_dir):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)
    test_dir = Path(base_dir) / 'data' / 'raw' / raw_test_dir
    
    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}. Please ensure test zip is extracted.")
        return
        
    logging.info(f"--- Generating Codabench Submission for {dataset_name} ---")
    
    # 1. Load FAISS Semantic Index
    logging.info("Loading Semantic Embeddings & FAISS Index...")
    embeddings = np.load(os.path.join(store_dir, 'article_embeddings.npy')).astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 2. Load Article mappings
    articles = pd.read_parquet(os.path.join(store_dir, 'articles.parquet'))
    id_to_idx = {str(aid): i for i, aid in enumerate(articles['article_id'])}
    
    # 3. Detect Dataset Type and Load Behaviors
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
        # For MIND, history is parsed directly from each row during iteration.
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
            
    total_rows = behaviors_test.height
    logging.info(f"Found {total_rows:,} test impressions.")
    
    output_file = os.path.join(base_dir, f"{raw_test_dir}_predictions.txt")
    
    with open(output_file, "w") as f:
        for count, row in enumerate(behaviors_test.iter_rows(named=True)):
            if count > 0 and count % 50000 == 0:
                logging.info(f"  Processed {count:,}/{total_rows:,} predictions...")
                
            imp_id = str(row['impression_id'])
            user_id = str(row['user_id'])
            
            if is_mind:
                history_str = row['history']
                history = history_str.split(" ") if history_str else []
                
                imp_str = row['impressions']
                candidates = imp_str.split(" ") if imp_str else []
            else:
                history = user_to_history.get(user_id, [])
                
                candidates = row.get('article_ids_inview', [])
                if isinstance(candidates, np.ndarray):
                    candidates = candidates.tolist()
                candidates = [str(c) for c in candidates]
                
            if not candidates:
                f.write(f"{imp_id} []\n")
                continue
                
            history_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
            
            if not history_indices:
                sem_scores = np.zeros(len(candidates))
            else:
                user_vector = np.mean(embeddings[history_indices], axis=0, keepdims=True)
                faiss.normalize_L2(user_vector)
                
                candidate_indices = [id_to_idx.get(aid, -1) for aid in candidates]
                
                sem_scores = []
                for idx in candidate_indices:
                    if idx == -1:
                        sem_scores.append(-999.0)
                    else:
                        cand_vec = embeddings[idx:idx+1]
                        sem_scores.append(np.dot(user_vector, cand_vec.T)[0][0])
                sem_scores = np.array(sem_scores)
                
            # Convert scores to dense ranks (1 is highest score)
            order = np.argsort(sem_scores)[::-1]
            ranks = np.zeros(len(sem_scores), dtype=int)
            ranks[order] = np.arange(1, len(sem_scores) + 1)
            
            f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
            
    # 4. Zip it up for Codabench
    zip_file = os.path.join(base_dir, f"{raw_test_dir}_predictions.zip")
    logging.info(f"Zipping {output_file} into {zip_file}...")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        arcname = "prediction.txt" if is_mind else "predictions.txt"
        zf.write(output_file, arcname=arcname)
        
    logging.info(f"Success! Ready for Codabench upload: {zip_file}")

if __name__ == '__main__':
    generate_submission('MINDlarge_train', 'MINDlarge_test')
    generate_submission('ebnerd_large', 'ebnerd_testset')
