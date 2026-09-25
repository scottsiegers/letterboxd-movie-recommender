import os
import sqlite3
import time

import pandas as pd

from src.tmdb import TMDBClient


DEFAULT_CATALOG_PATH = "data/movie_catalog.db"


class MovieCatalog:
    def __init__(
        self,
        tmdb_client: TMDBClient,
        db_path: str = DEFAULT_CATALOG_PATH,
    ):
        self.tmdb = tmdb_client
        self.db_path = db_path

        self._initialize_database()

    # ==================================================
    # Database setup
    # ==================================================

    def _initialize_database(self):
        os.makedirs(
            os.path.dirname(self.db_path),
            exist_ok=True,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog (
                    tmdb_id INTEGER PRIMARY KEY,
                    title TEXT,
                    original_title TEXT,
                    release_date TEXT,
                    year INTEGER,
                    overview TEXT,
                    genres TEXT,
                    director TEXT,
                    cast TEXT,
                    keywords TEXT,
                    runtime INTEGER,
                    popularity REAL,
                    vote_average REAL,
                    vote_count INTEGER,
                    poster_url TEXT
                )
                """
            )

    # ==================================================
    # Discover movies
    # ==================================================

    def discover_movies(
        self,
        page=1,
        sort_by="popularity.desc",
        min_vote_count=100,
        min_rating=5.0,
    ):
        """
        Retrieve one page of candidate movies from TMDB.
        """

        params = {
            "include_adult": False,
            "include_video": False,
            "language": "en-US",
            "page": page,
            "sort_by": sort_by,
            "vote_count.gte": min_vote_count,
            "vote_average.gte": min_rating,
        }

        data = self.tmdb._get(
            "/discover/movie",
            params=params,
        )

        return data.get("results", [])

    # ==================================================
    # Save movie
    # ==================================================

    def save_movie(self, movie):
        """
        Save enriched movie metadata into local catalog.
        """

        release_date = movie.get("release_date")

        year = None

        if release_date:
            try:
                year = int(release_date[:4])
            except (TypeError, ValueError):
                pass

        genres = movie.get("genres") or []
        cast = movie.get("cast") or []
        keywords = movie.get("keywords") or []

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO catalog (
                    tmdb_id,
                    title,
                    original_title,
                    release_date,
                    year,
                    overview,
                    genres,
                    director,
                    cast,
                    keywords,
                    runtime,
                    popularity,
                    vote_average,
                    vote_count,
                    poster_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    movie.get("tmdb_id"),
                    movie.get("tmdb_title"),
                    movie.get("original_title"),
                    release_date,
                    year,
                    movie.get("overview"),
                    "|".join(genres),
                    movie.get("director"),
                    "|".join(cast),
                    "|".join(keywords),
                    movie.get("runtime"),
                    movie.get("popularity"),
                    movie.get("vote_average"),
                    movie.get("vote_count"),
                    movie.get("poster_url"),
                ),
            )

    # ==================================================
    # Build catalog
    # ==================================================

    def build_catalog(
        self,
        pages=10,
        min_vote_count=100,
        min_rating=5.0,
        progress_callback=None,
    ):
        """
        Build candidate catalog from TMDB Discover.

        Each page usually returns around 20 movies.
        """

        total_movies_processed = 0

        for page in range(1, pages + 1):

            results = self.discover_movies(
                page=page,
                min_vote_count=min_vote_count,
                min_rating=min_rating,
            )

            for movie in results:

                tmdb_id = movie["id"]

                try:
                    metadata = self.tmdb.get_movie_metadata(
                        tmdb_id
                    )

                    self.save_movie(
                        metadata
                    )

                    total_movies_processed += 1

                except Exception as exc:
                    print(
                        f"Failed TMDB ID "
                        f"{tmdb_id}: {exc}"
                    )

                if progress_callback:
                    expected_total = pages * 20

                    progress_callback(
                        min(
                            total_movies_processed
                            / expected_total,
                            1.0,
                        )
                    )

                time.sleep(0.02)

        return total_movies_processed

    # ==================================================
    # Load catalog
    # ==================================================

    def load_catalog(self):
        """
        Return the entire candidate catalog as a DataFrame.
        """

        with sqlite3.connect(self.db_path) as conn:
            catalog = pd.read_sql_query(
                """
                SELECT *
                FROM catalog
                """,
                conn,
            )

        return catalog

    # ==================================================
    # Remove already watched movies
    # ==================================================

    def get_candidates(
        self,
        user_movies,
    ):
        """
        Return catalog movies the user has not already rated.
        """

        catalog = self.load_catalog()

        if "tmdb_id" not in user_movies.columns:
            raise ValueError(
                "user_movies must contain tmdb_id. "
                "Enrich the Letterboxd data with TMDB first."
            )

        watched_ids = set(
            user_movies["tmdb_id"]
            .dropna()
            .astype(int)
        )

        candidates = catalog[
            ~catalog["tmdb_id"].isin(
                watched_ids
            )
        ].copy()

        return candidates.reset_index(
            drop=True
        )