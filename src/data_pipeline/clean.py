import pandas as pd
import numpy as np
import polars as pl
import os
import glob
import logging

def process_mind(raw_dir, proc_dir):
    logging.info("Processing MIND dataset...")
    
    mind_dirs = glob.glob(os.path.join(raw_dir, 'MINDsmall*')) + glob.glob(os.path.join(raw_dir, 'MINDlarge*'))
    
    for mind_dir in mind_dirs:
        split_name = os.path.basename(mind_dir)
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
        articles = pl.read_csv(news_path, separator='\t', has_header=False, new_columns=news_cols, quote_char=None, truncate_ragged_lines=True)
        
        unified_articles = articles.select([
            pl.col('article_id').cast(pl.Utf8),
            pl.col('title').fill_null(""),
            pl.col('abstract').fill_null(""),
            pl.lit("").alias('body'),
            pl.col('category').fill_null(""),
            pl.lit(None, dtype=pl.Datetime).alias('published_time')
        ])
        unified_articles.write_parquet(os.path.join(out_dir, 'articles.parquet'))
        
        # 2. Behaviors & Users
        logging.info(f"  Parsing MIND Behaviors & Users for {split_name}...")
        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        behaviors = pl.read_csv(beh_path, separator='\t', has_header=False, new_columns=beh_cols, quote_char=None, truncate_ragged_lines=True)
        
        # Users (Click History)
        users = behaviors.group_by('user_id').agg(pl.col('history').first()).with_columns(
            pl.col('history').fill_null("").str.split(" ")
        ).select([
            pl.col('user_id').cast(pl.Utf8),
            pl.col('history').cast(pl.List(pl.Utf8))
        ])
        users.write_parquet(os.path.join(out_dir, 'users.parquet'))
        
        # Impressions
        def parse_mind_impressions(imp_str):
            if not imp_str: return {"clicked": [], "unclicked": []}
            c, u = [], []
            for item in str(imp_str).split():
                if '-' in item:
                    aid, click = item.split('-')
                    if click == '1': c.append(aid)
                    else: u.append(aid)
            return {"clicked": c, "unclicked": u}
            
        parsed = behaviors.select(pl.col("impressions").map_elements(parse_mind_impressions, return_dtype=pl.Struct([pl.Field("clicked", pl.List(pl.Utf8)), pl.Field("unclicked", pl.List(pl.Utf8))])).alias("parsed")).unnest("parsed")
        
        # Try multiple datetime formats for MIND since some are AM/PM and some are 24hr
        time_col = pl.coalesce([
            pl.col('time').str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False),
            pl.col('time').str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
        ])
        
        unified_impressions = behaviors.with_columns([
            parsed["clicked"].alias("clicked_articles"),
            parsed["unclicked"].alias("unclicked_articles")
        ]).select([
            pl.col('impression_id').cast(pl.Utf8),
            pl.col('user_id').cast(pl.Utf8),
            time_col.alias('time'),
            pl.col('clicked_articles'),
            pl.col('unclicked_articles')
        ])
        unified_impressions.write_parquet(os.path.join(out_dir, 'impressions.parquet'))


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
            articles = pl.read_parquet(articles_path)
            
            # Subtitle/body/category might be missing in some splits, handle gracefully
            if "subtitle" not in articles.columns:
                articles = articles.with_columns(pl.lit("").alias("subtitle"))
            if "body" not in articles.columns:
                articles = articles.with_columns(pl.lit("").alias("body"))
            if "category_str" not in articles.columns and "category" not in articles.columns:
                articles = articles.with_columns(pl.lit("").alias("category_str"))
            if "published_time" not in articles.columns:
                articles = articles.with_columns(pl.lit(None, dtype=pl.Datetime).alias("published_time"))

            cat_col = pl.col("category_str") if "category_str" in articles.columns else pl.col("category")

            unified_articles = articles.select([
                pl.col('article_id').cast(pl.Utf8),
                pl.col('title').fill_null(""),
                pl.col('subtitle').fill_null("").alias('abstract'),
                pl.col('body').fill_null(""),
                cat_col.fill_null("").alias('category'),
                pl.col('published_time')
            ])

            # articles_large_only.zip (new in the assignment's large-scale setup)
            # supplements ebnerd_large's own articles.parquet with additional
            # articles -- almost certainly the official fix for the same
            # test-period coverage gap generate_submission.py works around at
            # inference time. Schema isn't documented, so this merge is
            # best-effort: normalize with the same column fallbacks used
            # above, keep only article_ids not already present, and skip
            # cleanly (with a clear log line) if the file doesn't look like
            # we expect rather than crashing the pipeline.
            if split_name == 'ebnerd_large':
                extra_dir = os.path.join(os.path.dirname(ebnerd_dir), 'articles_large_only')
                if os.path.isdir(extra_dir):
                    extra_parquets = glob.glob(os.path.join(extra_dir, '**', '*.parquet'), recursive=True)
                    for extra_path in extra_parquets:
                        try:
                            extra = pl.read_parquet(extra_path)
                            if 'article_id' not in extra.columns or 'title' not in extra.columns:
                                logging.warning(f"  {extra_path} doesn't look like an article table (missing article_id/title) -- skipping.")
                                continue
                            extra_subtitle = pl.col('subtitle').fill_null("") if 'subtitle' in extra.columns else pl.lit("")
                            extra_body = pl.col('body').fill_null("") if 'body' in extra.columns else pl.lit("")
                            extra_cat = (pl.col('category_str') if 'category_str' in extra.columns
                                         else pl.col('category') if 'category' in extra.columns
                                         else pl.lit(""))
                            extra_published = pl.col('published_time') if 'published_time' in extra.columns else pl.lit(None, dtype=pl.Datetime)

                            extra_unified = extra.select([
                                pl.col('article_id').cast(pl.Utf8),
                                pl.col('title').fill_null(""),
                                extra_subtitle.alias('abstract'),
                                extra_body.alias('body'),
                                extra_cat.fill_null("").alias('category'),
                                extra_published.alias('published_time'),
                            ])

                            known_ids = unified_articles['article_id']
                            new_rows = extra_unified.filter(~pl.col('article_id').is_in(known_ids))
                            if new_rows.height > 0:
                                unified_articles = pl.concat([unified_articles, new_rows])
                                logging.info(f"  Merged {new_rows.height} additional articles from {extra_path} into ebnerd_large's catalog.")
                            else:
                                logging.info(f"  {extra_path} contributed no new article_ids beyond ebnerd_large's own catalog.")
                        except Exception as e:
                            logging.warning(f"  Could not merge {extra_path} into the article catalog ({e}) -- skipping it.")

            unified_articles.write_parquet(os.path.join(out_dir, 'articles.parquet'))
            
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
                
                # Use LazyFrame to prevent OOM
                unified_users = (
                    pl.scan_parquet(hist_path)
                    .select([
                        pl.col('user_id').cast(pl.Utf8),
                        pl.col('article_id_fixed').cast(pl.List(pl.Utf8)).alias('history')
                    ])
                    .unique(subset=['user_id'], maintain_order=False)
                    .collect(streaming=True)
                )
                unified_users.write_parquet(os.path.join(sub_out_dir, 'users.parquet'))
                
            # 3. Impressions
            beh_path = os.path.join(sub_dir, 'behaviors.parquet')
            if os.path.exists(beh_path):
                logging.info(f"  Parsing EB-NeRD Behaviors for {split_name}/{sub_split}...")
                
                unified_impressions = (
                    pl.scan_parquet(beh_path)
                    .select([
                        pl.col('impression_id').cast(pl.Utf8),
                        pl.col('user_id').cast(pl.Utf8),
                        pl.col('impression_time').alias('time'),
                        pl.col('article_ids_clicked').cast(pl.List(pl.Utf8)).alias('clicked_articles'),
                        pl.col('article_ids_inview').cast(pl.List(pl.Utf8)).list.set_difference(
                            pl.col('article_ids_clicked').cast(pl.List(pl.Utf8))
                        ).alias('unclicked_articles')
                    ])
                    .collect(streaming=True)
                )
                unified_impressions.write_parquet(os.path.join(sub_out_dir, 'impressions.parquet'))


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
