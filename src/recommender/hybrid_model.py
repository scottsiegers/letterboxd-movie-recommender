"""
LOCATION: src/recommender/hybrid_model.py

Hybrid recommender = SVD (trained OFFLINE) + TF-IDF content model.

The problem: a brand-new Letterboxd user isn't in the trained SVD, so
svd.predict() for them only returns global mean + movie bias (same ranking
for everybody). The fix is "fold-in": keep the learned movie factors fixed
and solve for this one user's bias + latent vector from their ratings.
It's a tiny ridge regression, so it takes milliseconds instead of retraining.
"""

import numpy as np
import pandas as pd


class SVDFoldIn:

    def __init__(self, mu, qi, bi, raw_ids, reg=0.2):
        self.reg = reg
        self.mu = float(mu)
        self.qi = qi
        self.bi = bi
        self.raw_ids = np.asarray(raw_ids, dtype=np.int64)
        self.inner_of = {int(r): i for i, r in enumerate(self.raw_ids)}

    @classmethod
    def from_npz(cls, path, reg=0.2):
        d = np.load(path)
        return cls(d["mu"], d["qi"], d["bi"], d["raw_ids"], reg)

    def score_user(self, user_ratings):
        """
        user_ratings: DataFrame with tmdbId, rating
        Returns predicted ratings for EVERY movie in the SVD, as a
        Series indexed by tmdbId.
        """

        df = user_ratings[["tmdbId", "rating"]].dropna().copy()
        df["tmdbId"] = df["tmdbId"].astype(np.int64)
        df = df.drop_duplicates(subset="tmdbId")
        df["inner"] = df["tmdbId"].map(self.inner_of)
        df = df.dropna(subset=["inner"])

        if len(df) < 3:
            raise ValueError(
                "Too few of the user's movies exist in the SVD model."
            )

        idx = df["inner"].astype(int).to_numpy()
        y = df["rating"].to_numpy(float) - self.mu - self.bi[idx]

        # Design matrix [1, q_i]  ->  solve for [b_u, p_u]
        A = np.hstack([np.ones((len(idx), 1)), self.qi[idx]])

        # Surprise's objective penalizes a user's parameters once per
        # rating, so the effective ridge strength is reg * n_ratings.
        lam = self.reg * len(idx)

        x = np.linalg.solve(
            A.T @ A + lam * np.eye(A.shape[1]),
            A.T @ y,
        )

        bu, pu = x[0], x[1:]

        preds = self.mu + bu + self.bi + self.qi @ pu

        return pd.Series(
            np.clip(preds, 0.5, 5.0),
            index=self.raw_ids,
        )


def hybrid_recommend(
    content,            # ContentRecommender
    svd_foldin,         # SVDFoldIn
    user_ratings,       # DataFrame: tmdbId, rating
    n_recs=10,
    min_votes=300,
    max_svd_weight=0.5,
    ratings_for_full_weight=50,
):
    """
    Blend = w * SVD percentile + (1 - w) * content percentile.

    w grows with how many ratings the user has (few ratings -> trust
    content more, many ratings -> trust collaborative filtering more).
    Movies the SVD has never seen (cold items) use content only.
    """

    # ---- content scores for every movie in the TF-IDF matrix ----------
    profile, rated_rows, weights = content.build_profile(user_ratings)
    content_scores = content.X.dot(profile) / np.linalg.norm(profile)

    out = pd.DataFrame({
        "tmdbId": content.tmdb_ids,
        "title": content.catalog["title"].to_numpy(),
        "content": content_scores,
    })

    # ---- poster info, if the catalog has it ---------------------------
    for col in ("poster_url", "poster_path"):
        if col in content.catalog.columns:
            out[col] = content.catalog[col].to_numpy()

    # ---- SVD predicted ratings, aligned to the same movies ------------
    svd_preds = svd_foldin.score_user(user_ratings)
    out["svd_rating"] = out["tmdbId"].map(svd_preds)

    # ---- eligibility: unrated + (optionally) not too obscure ----------
    eligible = np.ones(len(out), dtype=bool)
    eligible[rated_rows] = False

    if min_votes and "vote_count" in content.catalog.columns:
        eligible &= (
            content.catalog["vote_count"].fillna(0).to_numpy()
            >= min_votes
        )

    out = out[eligible].copy()

    # ---- percentile-normalize each score within the eligible pool -----
    out["content_pct"] = out["content"].rank(pct=True)
    out["svd_pct"] = out["svd_rating"].rank(pct=True)   # NaN stays NaN

    n_rated = len(user_ratings.dropna(subset=["tmdbId", "rating"]))
    w = max_svd_weight * min(1.0, n_rated / ratings_for_full_weight)

    has_svd = out["svd_pct"].notna()

    out["hybrid"] = out["content_pct"]
    out.loc[has_svd, "hybrid"] = (
        w * out.loc[has_svd, "svd_pct"]
        + (1 - w) * out.loc[has_svd, "content_pct"]
    )

    return (
        out.sort_values("hybrid", ascending=False)
        .head(n_recs)
        .reset_index(drop=True)
    )