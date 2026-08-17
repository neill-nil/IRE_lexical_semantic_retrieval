import argparse
import logging
from src.data_pipeline import extract, clean, split, feature_store

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

    # Step 4: Create Feature Store
    logging.info("--- Step 4: Building Feature Store ---")
    feature_store.run(dataset=args.dataset)

    logging.info("Pipeline completed successfully! All data is in the feature_store.")

if __name__ == "__main__":
    main()
