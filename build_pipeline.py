import argparse
import logging
from src.data_pipeline import extract, clean, split, feature_store, embed
from src.retrieval import bm25, train_ranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Build the data pipeline for Assignment 1.")
    parser.add_argument('--dataset', type=str, choices=['mind', 'ebnerd', 'both'], default='both',
                        help='Which dataset to process (mind, ebnerd, or both)')
    args = parser.parse_args()

    logging.info("Starting reproducible data pipeline...")

    # Step 1: Extract Zip Files
    logging.info("--- Step 1: Extracting raw data ---")
    extract.run(dataset=args.dataset)

    # Step 2: Clean and Parse Data
    logging.info("--- Step 2: Cleaning and parsing data ---")
    clean.run(dataset=args.dataset)

    # Step 3: Temporal Train/Val/Test Split
    logging.info("--- Step 3: Temporal splitting ---")
    split.run(dataset=args.dataset)
    
    # Step 4: Finalize Feature Store
    logging.info("--- Step 4: Finalizing Feature Store ---")
    feature_store.run(dataset=args.dataset)
    
    # Step 5: Compute Semantic Embeddings
    logging.info("--- Step 5: Computing Semantic Embeddings ---")
    embed.run(dataset=args.dataset)

    # Step 6: Build BM25 Inverted Indices
    logging.info("--- Step 6: Building BM25 Indices ---")
    bm25.run(dataset=args.dataset)

    # Step 7: Train Hybrid Ranker (BM25 + semantic + popularity + recency)
    logging.info("--- Step 7: Training Hybrid Ranker ---")
    train_ranker.run(dataset=args.dataset)

    logging.info("Data pipeline build complete!")

if __name__ == "__main__":
    main()
