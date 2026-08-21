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


def build_index_for_dataset(dataset_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


if __name__ == '__main__':
    datasets = ['MINDsmall_train', 'MINDsmall_dev', 'ebnerd_small', 'ebnerd_demo']
    for ds in datasets:
        build_index_for_dataset(ds)
