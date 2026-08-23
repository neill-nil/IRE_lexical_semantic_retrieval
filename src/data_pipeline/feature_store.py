import os
import glob
import logging
import shutil
import polars as pl

def run(dataset='both'):
    logging.info("Finalizing Feature Store...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    store_dir = os.path.join(base_dir, 'data', 'feature_store')
    
    datasets_to_process = []
    if dataset in ['mind', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDsmall*')))
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDlarge*')))
    if dataset in ['ebnerd', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'ebnerd*')))
        
    for d_path in datasets_to_process:
        dataset_name = os.path.basename(d_path)
        out_store = os.path.join(store_dir, dataset_name)
        
        logging.info(f"  Copying static features (Articles, Users) for {dataset_name} to feature store...")
        
        # 1. Articles (usually at the root of the processed dataset dir)
        articles_path = os.path.join(d_path, 'articles.parquet')
        if os.path.exists(articles_path):
            shutil.copy2(articles_path, os.path.join(out_store, 'articles.parquet'))
            
        # 2. Users (History). Might be at root or under subfolders depending on dataset.
        user_dfs = []
        user_path_root = os.path.join(d_path, 'users.parquet')
        if os.path.exists(user_path_root):
            user_dfs.append(user_path_root)
            
        for sub in ['train', 'validation']:
            user_path_sub = os.path.join(d_path, sub, 'users.parquet')
            if os.path.exists(user_path_sub):
                user_dfs.append(user_path_sub)
                
        if user_dfs:
            # Use LazyFrame to concat and drop duplicates efficiently
            merged_users = (
                pl.concat([pl.scan_parquet(p) for p in user_dfs])
                .unique(subset=['user_id'], keep='last', maintain_order=False)
                .collect(streaming=True)
            )
            merged_users.write_parquet(os.path.join(out_store, 'users.parquet'))
            
    logging.info("Feature Store is fully assembled and ready for candidate generation!")
