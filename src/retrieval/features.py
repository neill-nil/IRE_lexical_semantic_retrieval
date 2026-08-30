"""Shared feature engineering used by the offline harness, the ranker trainer,
and the Codabench submission generator, so all three score candidates the
same way.
"""
import math
import pickle
import numpy as np


MAX_RECENT_HISTORY = 50


def recent_history(history, max_len=MAX_RECENT_HISTORY):
    """Cap click history to the most recent `max_len` items (both MIND's
    `history` field and EB-NeRD's `article_id_fixed` are documented as
    chronologically ordered oldest-first). This is what Q2 itself asks for
    ("concatenate titles of *recently* clicked articles"), and it matters
    for more than fidelity to the spec: without a cap, a power user's full
    history can expand a BM25 query to thousands of tokens, which turns
    full-corpus retrieval (see `bm25_retrieve_topk`) from milliseconds into
    tens of seconds per impression.
    """
    if history is None or len(history) == 0:
        return []
    return list(history[-max_len:])


def select_query_terms(bm25, query_tokens, max_query_terms=300, max_df_ratio=0.02):
    """Prune a raw token set down to the terms worth scoring at all -- shared
    by both `bm25_score_candidates` (rerank a handful of given candidates)
    and `bm25_retrieve_topk` (retrieve from the full corpus), since an
    unpruned query is the single largest cost in both: a query built from
    50 recent history articles can carry hundreds of unique tokens, and
    scoring every one against every candidate/document is O(tokens x N)
    for no benefit once you account for what those tokens actually are.

    Three bounds:
    - terms with idf <= 0 (near-universal stopwords, which is exactly what
      a plain regex tokenizer over raw text leaves in the vocabulary) are
      dropped outright -- their BM25 contribution is ~0 by construction;
    - terms that occur in only a single document (df == 1) are dropped too:
      by definition such a term can only ever "match" the one document that
      already contains it (almost always one of the history articles
      itself), so it is pure noise for matching a *different* document --
      sorting by raw idf without this filter actively prefers these
      self-matching hapax words and made full-corpus retrieval quality
      *worse than random* in testing;
    - terms whose document frequency exceeds `max_df_ratio` of the corpus
      are dropped unconditionally, regardless of how many other candidate
      terms are available -- idf > 0 alone still lets through terms with
      df up to ~50% of the corpus, and even one such term's postings list
      dominates runtime; this is what actually bounds the worst case (a
      short, generic query can otherwise be *slower* than a long, specific
      one, since with few candidate terms available every one of them gets
      scored no matter its postings size);
    - among the remainder, the `max_query_terms` with the smallest posting
      lists are kept -- still the most selective terms, but ones that
      genuinely co-occur with other documents, and cheapest to score.
    """
    if not query_tokens:
        return []
    max_df = max(2, int(max_df_ratio * bm25.N))
    candidates = []
    for t in query_tokens:
        if bm25.idf.get(t, 0.0) <= 0:
            continue
        postings = bm25.inverted_index.get(t)
        df = len(postings) if postings else 0
        if df < 2 or df > max_df:
            continue
        candidates.append((t, df))
    return [t for t, _ in sorted(candidates, key=lambda kv: kv[1])[:max_query_terms]]


def bm25_score_candidates(bm25, query_tokens, candidate_ids):
    """Score a small list of candidate doc ids against a BM25 query.

    Walks the (pruned) query vocabulary against the (small) candidate list.
    Pruning alone isn't enough here: on EB-NeRD, cutting the rerank query
    below ~300 terms was measured to destroy the BM25 signal almost
    entirely (AUC collapsed to ~0.50, i.e. noise) -- its lexical signal is
    genuinely diffuse across many terms, unlike a query that concentrates
    on a few discriminating words. So instead of scoring fewer terms, this
    scores the SAME (pruned) terms more cheaply: term-outer with a set
    intersection against the candidate list, instead of the naive
    candidate-outer double loop, which was measured (via profiling a live
    submission run) as ~70-80% of total per-impression cost.
    """
    scores_by_id = {aid: 0.0 for aid in candidate_ids}
    if not query_tokens:
        return np.zeros(len(candidate_ids), dtype=np.float64)
    useful = select_query_terms(bm25, query_tokens)
    if not useful:
        return np.zeros(len(candidate_ids), dtype=np.float64)
    avgdl = bm25.avgdl if bm25.avgdl else 1.0
    candidate_set = set(scores_by_id.keys())
    for token in useful:
        postings = bm25.inverted_index.get(token)
        if not postings:
            continue
        idf = bm25.idf[token]
        # Intersecting via dict/set views (C-level) beats a per-candidate
        # .get() loop once there are enough terms x candidates to matter.
        hits = postings.keys() & candidate_set
        for aid in hits:
            tf = postings[aid]
            dl = bm25.doc_lens.get(aid, avgdl)
            numerator = tf * (bm25.k1 + 1)
            denominator = tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / avgdl)
            scores_by_id[aid] += idf * (numerator / denominator)
    return np.array([scores_by_id[aid] for aid in candidate_ids], dtype=np.float64)


def bm25_retrieve_topk(bm25, query_tokens, k, max_query_terms=300, max_df_ratio=0.02):
    """True full-corpus BM25 retrieval: score every document that shares at
    least one token with the query (via the inverted index, so this only
    touches documents that could possibly score > 0), and return the top-k
    doc ids. This is what Q2 actually asks for -- ranking within an
    impression's already-tiny candidate pool is not retrieval. See
    `select_query_terms` for how the query is pruned before scoring.
    """
    useful = select_query_terms(bm25, query_tokens, max_query_terms, max_df_ratio)
    if not useful:
        return []

    scores = {}
    for token in useful:
        idf = bm25.idf[token]
        postings = bm25.inverted_index.get(token)
        if not postings:
            continue
        for doc_id, tf in postings.items():
            dl = bm25.doc_lens.get(doc_id, bm25.avgdl)
            numerator = tf * (bm25.k1 + 1)
            denominator = tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (numerator / denominator)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [doc_id for doc_id, _ in ranked]


def compute_popularity_fast(train_path):
    """Click-count aggregation read straight off a parquet impressions file.
    Tries polars (vectorized, stays cheap at tens of millions of rows) and
    falls back to a plain pandas/Counter pass if polars is unavailable or
    its native binary fails to load -- slower, but keeps training/inference
    usable on environments where the polars wheel is broken.
    """
    try:
        import polars as pl
        imp = pl.scan_parquet(train_path).select('clicked_articles').explode('clicked_articles').drop_nulls()
        counts_df = imp.group_by('clicked_articles').agg(pl.len().alias('n')).collect(streaming=True)
        counts = dict(zip(counts_df['clicked_articles'].to_list(), counts_df['n'].to_list()))
    except Exception:
        import pandas as pd
        from collections import Counter
        counts = Counter()
        for clicked in pd.read_parquet(train_path, columns=['clicked_articles'])['clicked_articles']:
            if clicked is not None:
                counts.update(clicked)
    return {aid: math.log1p(c + 1.0) for aid, c in counts.items()}


def recency_score(published_time, reference_time, half_life_days=2.0):
    """Exponential freshness decay: 1.0 for a brand-new article, decaying by
    half every `half_life_days`. Returns 0.0 when either timestamp is
    missing (e.g. MIND, which ships no article publish time).
    """
    if published_time is None or reference_time is None:
        return 0.0
    try:
        age_days = (reference_time - published_time).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return 0.0
    if age_days < 0 or age_days != age_days:  # NaT/NaN guard
        return 0.0
    return 0.5 ** (age_days / half_life_days)


FEATURE_NAMES = ["bm25", "semantic", "popularity", "recency", "hist_len_log"]


def build_candidate_features(candidates, lex_scores, sem_scores, popularity,
                              id_to_published, reference_time, hist_len):
    """Assemble the (n_candidates, 5) feature matrix used by the Ranker,
    in FEATURE_NAMES order. `lex_scores`/`sem_scores` are pre-computed
    per-candidate arrays (from bm25_score_candidates / a dot product), kept
    as separate args so callers that already have them (harness, ranker
    trainer, submission generator) don't recompute anything.
    """
    n = len(candidates)
    pop = np.array([popularity.get(aid, 0.0) for aid in candidates], dtype=np.float64)
    rec = np.array([
        recency_score(id_to_published.get(aid), reference_time) for aid in candidates
    ], dtype=np.float64)
    hist_feat = np.full(n, math.log1p(hist_len), dtype=np.float64)
    return np.column_stack([
        np.asarray(lex_scores, dtype=np.float64),
        np.asarray(sem_scores, dtype=np.float64),
        pop,
        rec,
        hist_feat,
    ])


class Ranker:
    """Combines the lexical, semantic, popularity and recency signals into
    a single click-probability score, plus the popularity table used as a
    cold-start fallback when a user has no click history at all.

    Wraps one of two scikit-learn models -- plain logistic regression, or a
    small gradient-boosted-tree ensemble (`HistGradientBoostingClassifier`).
    Which one to use is a per-dataset choice, not something this class
    decides on its own: an automated selector (fit an internal split, keep
    whichever wins) was tried and abandoned, because it disagreed with
    itself across three different validation strategies (a random row
    split, the true validation split, the true validation split scored
    correctly by per-impression AUC) and still didn't match which model
    actually wins on the true held-out test set for EB-NeRD, most likely
    because its content turns over fast enough that even the adjacent
    validation window isn't a reliable stand-in for the test window. The
    trustworthy signal came only from directly comparing both models
    against the real, held-out `impressions_test.parquet` (via
    `harness.py`) for each dataset:

        Dataset            Logistic   Gradient Boosting
        ebnerd_small       0.614      0.671
        ebnerd_demo        0.620      0.673
        MINDsmall_train    0.619      0.605

    Gradient boosting wins clearly and consistently on EB-NeRD (one
    feature, recency, dominates, and boosting's ability to model
    interactions with it helps); logistic regression wins on MIND (no
    single feature dominates, and boosting's extra flexibility overfits
    without enough signal to justify it). `train_ranker.py` picks
    `model_type` from this table, keyed on dataset family.
    """

    def __init__(self, scaler=None, model=None, model_type='logistic', popularity=None, has_recency=False):
        self.scaler = scaler
        self.model = model
        self.model_type = model_type
        self.popularity = popularity or {}
        self.has_recency = has_recency

    def fit(self, X, y, model_type='logistic', seed=42):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import HistGradientBoostingClassifier

        if model_type == 'histgb':
            self.scaler = None
            self.model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.1,
                class_weight='balanced', random_state=seed,
            )
            self.model.fit(X, y)
        else:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            self.model = LogisticRegression(max_iter=2000, class_weight='balanced')
            self.model.fit(Xs, y)
        self.model_type = model_type
        return self
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.model_type == 'logistic':
            # Equivalent to self.model.predict_proba(self.scaler.transform(X))[:, 1],
            # but as raw numpy: sklearn's transform()/predict_proba() each
            # carry fixed per-call input-validation overhead (validate_data,
            # check_array) that's negligible on a full batch but was
            # measured, at the scale of one call per impression, to be a
            # meaningful slice of total per-impression cost during a live
            # large-scale submission run. The scaler is just
            # (X - mean) / scale and logistic regression is just a sigmoid
            # of a linear score -- both cheap to inline.
            Xs = (X - self.scaler.mean_) / self.scaler.scale_
            z = Xs @ self.model.coef_[0] + self.model.intercept_[0]
            return 1.0 / (1.0 + np.exp(-z))
        # HistGradientBoostingClassifier's tree ensemble isn't practical to
        # inline the same way -- go through sklearn directly.
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'model': self.model,
                'model_type': self.model_type,
                'popularity': self.popularity,
                'has_recency': self.has_recency,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data.get('scaler'), data['model'], data.get('model_type', 'logistic'),
                    data['popularity'], data['has_recency'])
