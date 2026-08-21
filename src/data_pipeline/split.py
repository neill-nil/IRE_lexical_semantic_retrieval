import pandas as pd
import os
import glob
import logging

def split_chronologically(df, max_time):
    # Test set: strictly the last 24 hours
    test_cutoff = max_time - pd.Timedelta(days=1)
    # Validation set: strictly the 24 hours before test set
    val_cutoff = max_time - pd.Timedelta(days=2)
    
    test_mask = df['time'] > test_cutoff
    val_mask = (df['time'] > val_cutoff) & (df['time'] <= test_cutoff)
    train_mask = df['time'] <= val_cutoff
    
    return df[train_mask], df[val_mask], df[test_mask]

def run(dataset='both'):
    logging.info("Starting chronological temporal splitting...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    store_dir = os.path.join(base_dir, 'data', 'feature_store')
    
    os.makedirs(store_dir, exist_ok=True)
    
    datasets_to_process = []
    if dataset in ['mind', 'both']:
        # Match MINDsmall_train and MINDsmall_dev processed folders
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'MINDsmall*')))
    if dataset in ['ebnerd', 'both']:
        datasets_to_process.extend(glob.glob(os.path.join(proc_dir, 'ebnerd*')))
        
    for d_path in datasets_to_process:
        dataset_name = os.path.basename(d_path)
        out_store = os.path.join(store_dir, dataset_name)
        os.makedirs(out_store, exist_ok=True)
        
        logging.info(f"  Splitting interactions for {dataset_name}...")
        
        # Load all available impressions for the dataset
        impressions_list = []
        
        # Check root level (like MIND)
        imp_file = os.path.join(d_path, 'impressions.parquet')
        if os.path.exists(imp_file):
            impressions_list.append(pd.read_parquet(imp_file))
            
        # Check subfolders (like EB-NeRD train/ validation/)
        for sub in ['train', 'validation']:
            sub_imp = os.path.join(d_path, sub, 'impressions.parquet')
            if os.path.exists(sub_imp):
                impressions_list.append(pd.read_parquet(sub_imp))
                
        if not impressions_list:
            logging.warning(f"  No impressions found for {dataset_name}. Skipping...")
            continue
            
        # Concatenate and sort chronologically
        df_imp = pd.concat(impressions_list, ignore_index=True)
        df_imp = df_imp.sort_values(by='time', ascending=True).reset_index(drop=True)
        
        max_time = df_imp['time'].max()
        train_df, val_df, test_df = split_chronologically(df_imp, max_time)
        
        logging.info(f"    Train size: {len(train_df)}")
        logging.info(f"    Val size:   {len(val_df)}")
        logging.info(f"    Test size:  {len(test_df)}")
        
        # Strict validation checks to ensure zero future-click leakage
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df['time'].max() <= val_df['time'].min(), "Leakage detected between Train and Val!"
        if len(val_df) > 0 and len(test_df) > 0:
            assert val_df['time'].max() <= test_df['time'].min(), "Leakage detected between Val and Test!"
        
        # Save to feature store
        train_df.to_parquet(os.path.join(out_store, 'impressions_train.parquet'))
        val_df.to_parquet(os.path.join(out_store, 'impressions_val.parquet'))
        test_df.to_parquet(os.path.join(out_store, 'impressions_test.parquet'))
        
    logging.info("Temporal splitting complete. Impressions saved to feature_store.")
