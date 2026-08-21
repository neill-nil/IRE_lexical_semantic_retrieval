import os
import glob
import logging
import shutil

def run(dataset='both'):
    logging.info("Finalizing Feature Store...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    store_dir = os.path.join(base_dir, 'data', 'feature_store')
    
    datasets_to_process = []
    if dataset in ['mind', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDsmall*')))
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
        # We will merge all unique user histories into one master users.parquet
        import pandas as pd
        user_dfs = []
        user_path_root = os.path.join(d_path, 'users.parquet')
        if os.path.exists(user_path_root):
            user_dfs.append(pd.read_parquet(user_path_root))
            
        for sub in ['train', 'validation']:
            user_path_sub = os.path.join(d_path, sub, 'users.parquet')
            if os.path.exists(user_path_sub):
                user_dfs.append(pd.read_parquet(user_path_sub))
                
        if user_dfs:
            merged_users = pd.concat(user_dfs, ignore_index=True)
            # Users might be duplicated across train/val splits in EB-NeRD, keep the one with longest history or just drop dupes.
            # To simplify, drop duplicates based on user_id keeping the first.
            merged_users = merged_users.drop_duplicates(subset=['user_id'], keep='last')
            merged_users.to_parquet(os.path.join(out_store, 'users.parquet'))
            
    logging.info("Feature Store is fully assembled and ready for candidate generation!")
