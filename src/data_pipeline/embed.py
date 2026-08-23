import os
import logging
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(dataset='both'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store')
    
    if dataset == 'both':
        datasets = [d for d in os.listdir(store_dir) if os.path.isdir(os.path.join(store_dir, d))]
    elif dataset == 'mind':
        datasets = [d for d in os.listdir(store_dir) if 'MIND' in d and os.path.isdir(os.path.join(store_dir, d))]
    elif dataset == 'ebnerd':
        datasets = [d for d in os.listdir(store_dir) if 'ebnerd' in d and os.path.isdir(os.path.join(store_dir, d))]
    else:
        datasets = [dataset]
        
    # Initialize the cross-lingual Siamese Network
    logging.info("Loading paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for ds in datasets:
        store_dir = os.path.join(base_dir, 'data', 'feature_store', ds)
        articles_path = os.path.join(store_dir, 'articles.parquet')
        
        if not os.path.exists(articles_path):
            logging.warning(f"No articles found for {ds} at {articles_path}")
            continue
            
        out_path = os.path.join(store_dir, 'article_embeddings.npy')
        # Skip if already computed to maintain idempotency
        if os.path.exists(out_path):
            logging.info(f"Embeddings already exist for {ds}. Skipping...")
            continue
            
        logging.info(f"--- Computing embeddings for {ds} ---")
        articles = pd.read_parquet(articles_path)
        
        # We replace NaNs with empty strings to prevent embedding crashes
        titles = articles['title'].fillna('')
        abstracts = articles['abstract'].fillna('')
        texts = (titles + " " + abstracts).tolist()
        
        logging.info(f"Passing {len(texts)} articles through the neural network...")
        # encode() automatically batches and leverages GPU if available
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        
        logging.info(f"Saving embeddings of shape {embeddings.shape} to {out_path}...")
        np.save(out_path, embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Semantic Embeddings.")
    parser.add_argument('--dataset', type=str, choices=['mind', 'ebnerd', 'both'], default='both',
                        help='Which dataset to process')
    args = parser.parse_args()
    run(dataset=args.dataset)
