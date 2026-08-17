import zipfile
import os
import glob
import logging

def run(dataset='both'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    # We look for zip files in the base directory
    zip_files = glob.glob(os.path.join(base_dir, '*.zip'))
    
    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        
        # Filter based on dataset arg
        if dataset == 'mind' and 'ebnerd' in filename.lower():
            continue
        if dataset == 'ebnerd' and 'mind' in filename.lower():
            continue
            
        # Do not extract embeddings right now if they are not finished
        if 'word2vec' in filename.lower() or 'bert' in filename.lower() or 'roberta' in filename.lower():
            continue
            
        logging.info(f"Extracting {filename} into {data_dir}...")
        
        # Create a specific folder for this zip to avoid overlapping
        extract_to = os.path.join(data_dir, filename.replace('.zip', ''))
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
    logging.info("Extraction complete.")
