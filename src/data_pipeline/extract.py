import zipfile
import os
import glob
import logging

def run(dataset='both'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'raw')

    # Only match known raw-dataset archive name patterns -- NOT a blind
    # glob of every *.zip in the repo root. This repo root is also where
    # things like kaggle_bundle.zip and *_predictions.zip submission
    # outputs live, and a blind glob previously deleted them (extract()
    # unlinks the zip after extracting it).
    dataset_patterns = ['MINDsmall*.zip', 'MINDlarge*.zip', 'ebnerd_demo*.zip',
                         'ebnerd_small*.zip', 'ebnerd_large*.zip', 'ebnerd_testset*.zip',
                         'articles_large_only*.zip']
    zip_files = []
    for pattern in dataset_patterns:
        zip_files.extend(glob.glob(os.path.join(base_dir, pattern)))

    # articles_large_only.zip supplements the EB-NeRD large article catalog --
    # doesn't have "ebnerd" in its filename, so it needs an explicit mind/ebnerd
    # filter of its own.
    for zip_path in zip_files:
        filename = os.path.basename(zip_path)

        # Filter based on dataset arg
        is_ebnerd_only = 'ebnerd' in filename.lower() or filename.lower().startswith('articles_large_only')
        if dataset == 'mind' and is_ebnerd_only:
            continue
        if dataset == 'ebnerd' and 'mind' in filename.lower():
            continue

        # Do not extract embeddings right now if they are not finished
        if 'word2vec' in filename.lower() or 'bert' in filename.lower() or 'roberta' in filename.lower():
            continue
            
        extract_to = os.path.join(data_dir, filename.replace('.zip', ''))
        
        # Skip if already extracted
        if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
            logging.info(f"Skipping {filename}, already extracted at {extract_to}")
            continue
            
        os.makedirs(extract_to, exist_ok=True)
        
        logging.info(f"Extracting {filename} into {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        logging.info(f"Deleting {filename} to free up disk space...")
        os.remove(zip_path)
            
    logging.info("Extraction complete.")
