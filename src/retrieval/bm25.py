import os
import re
import math
import pickle
import logging
from collections import defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0
        self.doc_lens = {}
        self.idf = {}
        # inverted_index[term][doc_id] = term_frequency
        self.inverted_index = defaultdict(dict)
        
    def tokenize(self, text):
        """Simple, robust regex tokenizer for English and Danish."""
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus_dict):
        """
        Build the inverted index and compute BM25 statistics.
        corpus_dict: {article_id: text}
        """
        self.N = len(corpus_dict)
        total_length = 0
        
        # Document frequencies for IDF calculation
        doc_freqs = defaultdict(int)
        
        logging.info(f"Tokenizing and indexing {self.N} documents...")
        for count, (doc_id, text) in enumerate(corpus_dict.items()):
            if count > 0 and count % 20000 == 0:
                logging.info(f"  Indexed {count}/{self.N} documents...")
                
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            self.doc_lens[doc_id] = doc_len
            total_length += doc_len
            
            # Compute term frequencies for this document
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
                
            # Update inverted index and document frequencies
            for token, tf in term_freqs.items():
                self.inverted_index[token][doc_id] = tf
                doc_freqs[token] += 1
                
        self.avgdl = total_length / self.N if self.N > 0 else 0
        
        logging.info("Computing IDF scores...")
        for term, df in doc_freqs.items():
            # Standard BM25 IDF formula
            idf_score = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            self.idf[term] = idf_score
            
        # Convert defaultdict to standard dict for pickling
        self.inverted_index = dict(self.inverted_index)
        logging.info(f"Index built successfully! Vocabulary size: {len(self.inverted_index)}")

    def save(self, path):
        logging.info(f"Saving BM25 index to {path}...")
        with open(path, 'wb') as f:
            pickle.dump({
                'k1': self.k1,
                'b': self.b,
                'N': self.N,
                'avgdl': self.avgdl,
                'doc_lens': self.doc_lens,
                'idf': self.idf,
                'inverted_index': self.inverted_index
            }, f)

    def add_documents(self, corpus_dict):
        """In-memory-only incorporation of documents the persisted index was
        never built with (e.g. articles that only exist in a test-period
        bundle). Skips ids already indexed. IDF is recomputed exactly for
        tokens touched by the new documents; IDF for untouched tokens is
        left as-is (a second-order approximation, since N technically grew
        for them too) -- fine for scoring a handful of otherwise-unscored
        candidates, not meant to replace rebuilding the index from scratch.
        Do not call save() after this -- the update is not persisted.
        """
        added_lengths = []
        touched_tokens = set()
        for doc_id, text in corpus_dict.items():
            if doc_id in self.doc_lens:
                continue
            tokens = self.tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            added_lengths.append(len(tokens))
            term_freqs = defaultdict(int)
            for tok in tokens:
                term_freqs[tok] += 1
            for tok, tf in term_freqs.items():
                self.inverted_index.setdefault(tok, {})[doc_id] = tf
                touched_tokens.add(tok)

        if not added_lengths:
            return 0

        total_len_before = self.avgdl * self.N
        self.N += len(added_lengths)
        self.avgdl = (total_len_before + sum(added_lengths)) / self.N
        for tok in touched_tokens:
            df = len(self.inverted_index[tok])
            self.idf[tok] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        return len(added_lengths)

    @classmethod
    def load(cls, path):
        logging.info(f"Loading BM25 index from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        index = cls(k1=data['k1'], b=data['b'])
        index.N = data['N']
        index.avgdl = data['avgdl']
        index.doc_lens = data['doc_lens']
        index.idf = data['idf']
        index.inverted_index = data['inverted_index']
        return index


def build_index_for_dataset(dataset_name, base_dir=None):
    base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store_dir = os.path.join(base_dir, 'data', 'feature_store', dataset_name)

    articles_path = os.path.join(store_dir, 'articles.parquet')
    if not os.path.exists(articles_path):
        logging.warning(f"No articles found for {dataset_name} at {articles_path}")
        return

    logging.info(f"Loading articles for {dataset_name}...")
    articles = pd.read_parquet(articles_path)

    # Create concatenated text corpus (title + abstract)
    articles['full_text'] = articles['title'].fillna('') + " " + articles['abstract'].fillna('')
    corpus_dict = dict(zip(articles['article_id'], articles['full_text']))

    # Build Index
    bm25 = BM25Index()
    bm25.fit(corpus_dict)

    # Save Index
    bm25.save(os.path.join(store_dir, 'bm25_index.pkl'))


def run(dataset='both'):
    """Build BM25 indices for every feature-store dataset matching the
    --dataset filter. Mirrors embed.run()'s dynamic directory scan so new
    datasets (e.g. MINDlarge_test, ebnerd_large) don't need to be
    hardcoded here to be picked up by build_pipeline.py.
    """
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
        index_path = os.path.join(store_root, ds, 'bm25_index.pkl')
        if os.path.exists(index_path):
            logging.info(f"BM25 index already exists for {ds}. Skipping...")
            continue
        logging.info(f"--- Building BM25 index for {ds} ---")
        build_index_for_dataset(ds, base_dir=base_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Build BM25 inverted indices.")
    parser.add_argument('--dataset', type=str, choices=['mind', 'ebnerd', 'both'], default='both')
    args = parser.parse_args()
    run(dataset=args.dataset)
