import pandas as pd
import numpy as np
import os
import glob
import logging

def parse_mind_impressions(imp_str):
    if pd.isna(imp_str):
        return [], []
    clicked = []
    unclicked = []
    for item in str(imp_str).split():
        if '-' in item:
            article_id, click = item.split('-')
            if click == '1':
                clicked.append(article_id)
            else:
                unclicked.append(article_id)
    return clicked, unclicked

def process_mind(raw_dir, proc_dir):
    logging.info("Processing MIND dataset...")
    
    # We look for MINDsmall_train and MINDsmall_dev
    mind_dirs = glob.glob(os.path.join(raw_dir, 'MINDsmall*'))
    
    for mind_dir in mind_dirs:
        split_name = os.path.basename(mind_dir)
        # Handle the double directory structure from extraction
        target_dir = os.path.join(mind_dir, split_name) if os.path.exists(os.path.join(mind_dir, split_name)) else mind_dir
        
        news_path = os.path.join(target_dir, 'news.tsv')
        beh_path = os.path.join(target_dir, 'behaviors.tsv')
        
        if not os.path.exists(news_path) or not os.path.exists(beh_path):
            continue
            
        out_dir = os.path.join(proc_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Articles
        logging.info(f"  Parsing MIND Articles for {split_name}...")
        news_cols = ['article_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        articles = pd.read_csv(news_path, sep='\t', names=news_cols)
        
        unified_articles = pd.DataFrame({
            'article_id': articles['article_id'],
            'title': articles['title'].fillna(''),
            'abstract': articles['abstract'].fillna(''),
            'body': '',  # MIND small typically lacks full body text
            'category': articles['category'].fillna('')
        })
        unified_articles.to_parquet(os.path.join(out_dir, 'articles.parquet'))
        
        # 2. Behaviors & Users
        logging.info(f"  Parsing MIND Behaviors & Users for {split_name}...")
        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        behaviors = pd.read_csv(beh_path, sep='\t', names=beh_cols)
        
        # Users (Click History)
        users = behaviors.groupby('user_id')['history'].first().reset_index()
        users['history'] = users['history'].apply(lambda x: str(x).split() if pd.notna(x) else [])
        users.to_parquet(os.path.join(out_dir, 'users.parquet'))
        
        # Impressions
        behaviors['time'] = pd.to_datetime(behaviors['time'])
        
        # Parse impressions into clicked and unclicked lists
        parsed = behaviors['impressions'].apply(parse_mind_impressions)
        behaviors['clicked_articles'] = parsed.apply(lambda x: x[0])
        behaviors['unclicked_articles'] = parsed.apply(lambda x: x[1])
        
        unified_impressions = pd.DataFrame({
            'impression_id': behaviors['impression_id'].astype(str),
            'user_id': behaviors['user_id'],
            'time': behaviors['time'],
            'clicked_articles': behaviors['clicked_articles'],
            'unclicked_articles': behaviors['unclicked_articles']
        })
        unified_impressions.to_parquet(os.path.join(out_dir, 'impressions.parquet'))


def process_ebnerd(raw_dir, proc_dir):
    logging.info("Processing EB-NeRD dataset...")
    
    ebnerd_dirs = glob.glob(os.path.join(raw_dir, 'ebnerd*'))
    
    for ebnerd_dir in ebnerd_dirs:
        split_name = os.path.basename(ebnerd_dir)
        out_dir = os.path.join(proc_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        
        articles_path = os.path.join(ebnerd_dir, 'articles.parquet')
        
        # 1. Articles
        if os.path.exists(articles_path):
            logging.info(f"  Parsing EB-NeRD Articles for {split_name}...")
            articles = pd.read_parquet(articles_path)
            unified_articles = pd.DataFrame({
                'article_id': articles['article_id'].astype(str),
                'title': articles['title'].fillna(''),
                'abstract': articles.get('subtitle', pd.Series(['']*len(articles))).fillna(''),
                'body': articles.get('body', pd.Series(['']*len(articles))).fillna(''),
                'category': articles.get('category_str', articles.get('category', pd.Series(['']*len(articles)))).fillna('')
            })
            unified_articles.to_parquet(os.path.join(out_dir, 'articles.parquet'))
            
        # EB-NeRD has subfolders like train/ and validation/ inside the zip
        for sub_split in ['train', 'validation']:
            sub_dir = os.path.join(ebnerd_dir, sub_split)
            if not os.path.exists(sub_dir):
                continue
                
            sub_out_dir = os.path.join(out_dir, sub_split)
            os.makedirs(sub_out_dir, exist_ok=True)
            
            # 2. Users (History)
            hist_path = os.path.join(sub_dir, 'history.parquet')
            if os.path.exists(hist_path):
                logging.info(f"  Parsing EB-NeRD History for {split_name}/{sub_split}...")
                history = pd.read_parquet(hist_path)
                unified_users = pd.DataFrame({
                    'user_id': history['user_id'].astype(str),
                    'history': history['article_id_fixed'].apply(lambda x: [str(i) for i in x] if x is not None else [])
                })
                # Drop duplicates if multiple rows per user exist
                unified_users = unified_users.drop_duplicates(subset=['user_id'])
                unified_users.to_parquet(os.path.join(sub_out_dir, 'users.parquet'))
                
            # 3. Impressions
            beh_path = os.path.join(sub_dir, 'behaviors.parquet')
            if os.path.exists(beh_path):
                logging.info(f"  Parsing EB-NeRD Behaviors for {split_name}/{sub_split}...")
                behaviors = pd.read_parquet(beh_path)
                
                # Compute unclicked by doing set difference: inview - clicked

                
                # actually:
                def safe_str_list(arr):
                    if arr is None: return []
                    if isinstance(arr, np.ndarray): return [str(x) for x in arr]
                    return [str(x) for x in arr]
                    
                unified_impressions = pd.DataFrame({
                    'impression_id': behaviors['impression_id'].astype(str),
                    'user_id': behaviors['user_id'].astype(str),
                    'time': pd.to_datetime(behaviors['impression_time']),
                    'clicked_articles': behaviors['article_ids_clicked'].apply(safe_str_list),
                })
                
                inview_series = behaviors['article_ids_inview'].apply(safe_str_list)
                unified_impressions['unclicked_articles'] = [
                    [i for i in inv if i not in set(clk)] 
                    for inv, clk in zip(inview_series, unified_impressions['clicked_articles'])
                ]
                
                unified_impressions.to_parquet(os.path.join(sub_out_dir, 'impressions.parquet'))


def run(dataset='both'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(proc_dir, exist_ok=True)
    
    if dataset in ['mind', 'both']:
        process_mind(raw_dir, proc_dir)
        
    if dataset in ['ebnerd', 'both']:
        process_ebnerd(raw_dir, proc_dir)
        
    logging.info("Data cleaning and parsing into unified schema is complete.")
