# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on MIND and EB-NeRD

A reproducible pipeline that ranks candidate news articles by click likelihood,
combining BM25 lexical retrieval, embedding-based semantic retrieval, article
popularity, and freshness into a single hybrid ranker, evaluated with an
offline harness (AUC/MRR/nDCG, diversity, cold/warm slicing, bootstrap CIs)
and submitted to both Codabench leaderboards (MIND, RecSys 2024 / EB-NeRD).

## One-command rebuild

```bash
pip install -r requirements.txt

# Small-scale, local (MINDsmall + ebnerd_demo/small) -- what CI/graders should run
python build_pipeline.py --dataset both
```

This single command runs the full pipeline end to end:
1. **Extract** — unzips any dataset archives found in the repo root into `data/raw/`.
2. **Clean** — parses MIND/EB-NeRD raw files into a unified schema (`articles`, `users`/history, `impressions`).
3. **Split** — chronological (never random) train/val/test split per dataset, with an assertion that no split's time range leaks into an earlier one.
4. **Feature store** — assembles per-dataset article/user tables under `data/feature_store/<dataset>/`.
5. **Embed** — computes sentence-transformer article embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).
6. **BM25** — builds an inverted index per dataset.
7. **Train ranker** — fits a logistic-regression hybrid ranker (BM25 + semantic + popularity + recency) on a sample of each dataset's train impressions.

Run `--dataset mind` or `--dataset ebnerd` to build just one side.

## Evaluate

```bash
# Full Q2/Q3 recall@K (BM25 and semantic retrieve top-200 from the FULL article corpus, not just the impression's own candidates)
PYTHONPATH=. python -m src.retrieval.eval_bm25 --dataset ebnerd_small
PYTHONPATH=. python -m src.retrieval.eval_semantic --dataset ebnerd_small

# Q4/Q5 offline harness: AUC/MRR/nDCG@5/@10, ILD@5, cold-vs-warm slicing, bootstrap 95% CIs
# (adds a third "Hybrid" column automatically once ranker.pkl exists for the dataset)
PYTHONPATH=. python -m src.evaluation.harness --dataset ebnerd_small
```

Metrics are written to `data/feature_store/<dataset>/metrics_*.json`.

## Generate a Codabench submission (large-scale)

The large datasets (MINDlarge, ebnerd_large, ebnerd_testset) don't fit this
repo's local dev environment — build and run on Kaggle instead. See
[KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md) for the exact steps. In short:

```bash
PYTHONPATH=. python build_pipeline.py --dataset both     # on the large raw data
PYTHONPATH=. python -m src.evaluation.generate_submission --dataset both
```

This writes `MINDlarge_test_predictions.zip` and `ebnerd_testset_predictions.zip`,
ready to upload to the two Codabench competitions.

## Design notes / known limitations

- **Recency** is only available for EB-NeRD (articles carry `published_time`); MIND's `news.tsv` has no publish timestamp, so the recency feature is always 0 there — the ranker still uses BM25/semantic/popularity for MIND.
- **BM25 lexical score for test-period-only articles** is computed by incorporating those articles into the loaded index in memory (`BM25Index.add_documents`); this is not persisted back to `bm25_index.pkl`, so a full index rebuild is still the source of truth if you want term stats to include the test period permanently.
- The hybrid ranker is trained on a bounded sample of train impressions (`--sample-size`, default 200k) — fitting itself is instant, feature extraction is the bottleneck at MINDlarge/ebnerd_large scale.
- See [ai_usage_log.md](ai_usage_log.md) for the AI-assistance disclosure.
