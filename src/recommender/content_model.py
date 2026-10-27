"""
LOCATION: src/recommender/content_model.py

Content-based recommender built on a precomputed TF-IDF matrix
(one row per movie, built from TMDB metadata).

How it works
------------
1. Take the user's rated movies and look up their rows in the TF-IDF matrix.
2. Build a taste profile = weighted sum of those rows, where the weight is
   the user's rating centered on their own average (so movies they rated
   above their average pull the profile toward them, and movies they rated
   below their average push it away).
3. Cosine-similarity of the profile against every movie -> score.
4. Drop movies the user already rated, return the top N.

There is nothing to train per user, so this runs in milliseconds.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class ContentRecommender:

    def __init__(self, tfidf_matrix, tmdb_ids, movie_catalog):
        """
        tfidf_matrix  : scipy sparse matrix, shape (n_movies, n_features)
        tmdb_ids      : sequence of tmdbIds, SAME ORDER as the matrix rows
        movie_catalog : DataFrame with at least tmdbId and title
                        (optional: vote_count, poster_url)
        """

        if tfidf_matrix.shape[0] != len(tmdb_ids):
            raise ValueError(
                f"Matrix has {tfidf_matrix.shape[0]} rows but "
                f"{len(tmdb_ids)} tmdbIds were provided. They must align."
            )

        # L2-normalize rows so a dot product == cosine similarity
        self.X = normalize(tfidf_matrix.tocsr())

        self.tmdb_ids = np.asarray(tmdb_ids).astype(np.int64)

        self.row_of = {
            int(tmdb_id): row
            for row, tmdb_id in enumerate(self.tmdb_ids)
        }

        # Catalog metadata lined up with matrix rows
        self.catalog = (
            movie_catalog
            .drop_duplicates(subset="tmdbId")
            .set_index("tmdbId")
            .reindex(self.tmdb_ids)
        )

    # ------------------------------------------------------------------
    def build_profile(self, user_ratings, center=True):
        """
        user_ratings: DataFrame with columns tmdbId, rating
        Returns (profile_vector, rows, weights)
        """

        df = (
            user_ratings[["tmdbId", "rating"]]
            .dropna()
            .copy()
        )

        df["tmdbId"] = df["tmdbId"].astype(np.int64)
        df = df.drop_duplicates(subset="tmdbId")

        df["row"] = df["tmdbId"].map(self.row_of)
        df = df.dropna(subset=["row"])

        if df.empty:
            raise ValueError(
                "None of the user's rated movies were found in the "
                "TF-IDF matrix."
            )

        rows = df["row"].astype(int).to_numpy()
        ratings = df["rating"].to_numpy(dtype=float)

        weights = ratings - ratings.mean() if center else ratings

        # User rated everything the same -> centering zeroes everything out
        if np.allclose(weights, 0):
            weights = ratings

        profile = self.X[rows].T.dot(weights)

        if np.linalg.norm(profile) == 0:
            raise ValueError("Could not build a taste profile.")

        return profile, rows, weights

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_ratings,
        n_recs=10,
        center=True,
        min_votes=0,
        explain=True,
    ):

        profile, rated_rows, weights = self.build_profile(
            user_ratings,
            center=center,
        )

        scores = self.X.dot(profile) / np.linalg.norm(profile)

        # Never recommend what they've already rated
        scores[rated_rows] = -np.inf

        # Optional: filter out obscure movies (very common with 87k titles)
        if min_votes and "vote_count" in self.catalog.columns:
            too_obscure = (
                self.catalog["vote_count"]
                .fillna(0)
                .to_numpy() < min_votes
            )
            scores[too_obscure] = -np.inf

        n_recs = min(n_recs, int(np.isfinite(scores).sum()))

        top = np.argpartition(-scores, n_recs - 1)[:n_recs]
        top = top[np.argsort(-scores[top])]

        recs = pd.DataFrame({
            "tmdbId": self.tmdb_ids[top],
            "title": self.catalog["title"].to_numpy()[top],
            "similarity": scores[top],
        })

        if "poster_url" in self.catalog.columns:
            recs["poster_url"] = (
                self.catalog["poster_url"].to_numpy()[top]
            )

        # "Because you liked ..." -> the liked movie most similar to each rec
        if explain:
            liked = weights > 0

            if liked.any():
                liked_rows = rated_rows[liked]
                liked_w = weights[liked]

                sims = self.X[top].dot(self.X[liked_rows].T).toarray()
                best = np.argmax(sims * liked_w, axis=1)

                recs["because_you_liked"] = (
                    self.catalog["title"]
                    .to_numpy()[liked_rows[best]]
                )

        return recs.reset_index(drop=True)