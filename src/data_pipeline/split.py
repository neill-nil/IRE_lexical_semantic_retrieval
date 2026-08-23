import polars as pl
import os
import glob
import logging
from datetime import timedelta

def split_chronologically(df_lazy, max_time):
    # Test set: strictly the last 24 hours
    test_cutoff = max_time - timedelta(days=1)
    # Validation set: strictly the 24 hours before test set
    val_cutoff = max_time - timedelta(days=2)
    
    train_lazy = df_lazy.filter(pl.col('time') <= val_cutoff)
    val_lazy = df_lazy.filter((pl.col('time') > val_cutoff) & (pl.col('time') <= test_cutoff))
    test_lazy = df_lazy.filter(pl.col('time') > test_cutoff)
    
    return train_lazy, val_lazy, test_lazy

def run(dataset='both'):
    logging.info("Starting chronological temporal splitting...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    store_dir = os.path.join(base_dir, 'data', 'feature_store')
    
    os.makedirs(store_dir, exist_ok=True)
    
    datasets_to_process = []
    if dataset in ['mind', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDsmall*')))
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDlarge*')))
    if dataset in ['ebnerd', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'ebnerd*')))
        
    for d_path in datasets_to_process:
        dataset_name = os.path.basename(d_path)
        out_store = os.path.join(store_dir, dataset_name)
        os.makedirs(out_store, exist_ok=True)
        
        logging.info(f"  Splitting interactions for {dataset_name}...")
        
        impressions_list = []
        
        imp_file = os.path.join(d_path, 'impressions.parquet')
        if os.path.exists(imp_file):
            impressions_list.append(imp_file)
            
        for sub in ['train', 'validation']:
            sub_imp = os.path.join(d_path, sub, 'impressions.parquet')
            if os.path.exists(sub_imp):
                impressions_list.append(sub_imp)
                
        if not impressions_list:
            logging.warning(f"  No impressions found for {dataset_name}. Skipping...")
            continue
            
        # 1. Concat and Sort via LazyFrame
        df_lazy = pl.concat([pl.scan_parquet(p) for p in impressions_list]).sort('time')
        
        # 2. Get the maximum time
        max_time = df_lazy.select(pl.col('time').max()).collect().item()
        
        if max_time is None:
            continue
            
        train_lazy, val_lazy, test_lazy = split_chronologically(df_lazy, max_time)
        
        # 3. Collect (execute query) and Write
        # Using streaming=True keeps memory usage extremely low
        train_df = train_lazy.collect(streaming=True)
        val_df = val_lazy.collect(streaming=True)
        test_df = test_lazy.collect(streaming=True)
        
        logging.info(f"    Train size: {train_df.height}")
        logging.info(f"    Val size:   {val_df.height}")
        logging.info(f"    Test size:  {test_df.height}")
        
        # Strict validation checks to ensure zero future-click leakage
        if train_df.height > 0 and val_df.height > 0:
            assert train_df['time'].max() <= val_df['time'].min(), "Leakage detected between Train and Val!"
        if val_df.height > 0 and test_df.height > 0:
            assert val_df['time'].max() <= test_df['time'].min(), "Leakage detected between Val and Test!"
        
        train_df.write_parquet(os.path.join(out_store, 'impressions_train.parquet'))
        val_df.write_parquet(os.path.join(out_store, 'impressions_val.parquet'))
        test_df.write_parquet(os.path.join(out_store, 'impressions_test.parquet'))
        
    logging.info("Temporal splitting complete. Impressions saved to feature_store.")
