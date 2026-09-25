import json
import os
import sqlite3
import time
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

DEFAULT_CACHE_PATH = "data/tmdb_cache.db"


class TMDBClient:
    def __init__(
        self,
        token: Optional[str] = None,
        cache_path: str = DEFAULT_CACHE_PATH,
    ):
        self.token = token or os.getenv("TMDB_TOKEN")

        if not self.token:
            raise ValueError(
                "TMDB token not found. "
                "Set TMDB_TOKEN in your environment or .env file."
            )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "accept": "application/json",
            }
        )

        self.cache_path = cache_path

        self._initialize_cache()

    # ==================================================
    # Cache setup
    # ==================================================

    def _initialize_cache(self):
        os.makedirs(
            os.path.dirname(self.cache_path),
            exist_ok=True,
        )

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS movies (
                    tmdb_id INTEGER PRIMARY KEY,
                    title TEXT,
                    year INTEGER,
                    metadata_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS movie_matches (
                    letterboxd_title TEXT NOT NULL,
                    letterboxd_year INTEGER,
                    tmdb_id INTEGER,
                    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (
                        letterboxd_title,
                        letterboxd_year
                    )
                )
                """
            )

    # ==================================================
    # Generic TMDB request
    # ==================================================

    def _get(self, endpoint, params=None):
        url = f"{TMDB_BASE_URL}{endpoint}"

        response = self.session.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    # ==================================================
    # Cache helpers
    # ==================================================

    def get_cached_metadata(self, tmdb_id):
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                """
                SELECT metadata_json
                FROM movies
                WHERE tmdb_id = ?
                """,
                (int(tmdb_id),),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def save_metadata_to_cache(self, metadata):
        tmdb_id = metadata["tmdb_id"]

        release_date = metadata.get("release_date")

        year = None

        if release_date:
            try:
                year = int(release_date[:4])
            except (TypeError, ValueError):
                pass

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO movies (
                    tmdb_id,
                    title,
                    year,
                    metadata_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    tmdb_id,
                    metadata.get("tmdb_title"),
                    year,
                    json.dumps(metadata),
                ),
            )

    def get_cached_match(self, title, year):
        normalized_title = str(title).strip().lower()

        year_value = None if year is None else int(year)

        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                """
                SELECT tmdb_id
                FROM movie_matches
                WHERE letterboxd_title = ?
                AND (
                    letterboxd_year = ?
                    OR (
                        letterboxd_year IS NULL
                        AND ? IS NULL
                    )
                )
                """,
                (
                    normalized_title,
                    year_value,
                    year_value,
                ),
            ).fetchone()

        if row is None:
            return None

        return row[0]

    def save_match_to_cache(
        self,
        title,
        year,
        tmdb_id,
    ):
        normalized_title = str(title).strip().lower()

        year_value = None if year is None else int(year)

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO movie_matches (
                    letterboxd_title,
                    letterboxd_year,
                    tmdb_id,
                    matched_at
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    normalized_title,
                    year_value,
                    tmdb_id,
                ),
            )

    # ==================================================
    # Search
    # ==================================================

    def search_movie(self, title, year=None):
        params = {
            "query": title,
            "include_adult": False,
            "language": "en-US",
        }

        if year:
            params["year"] = int(year)

        data = self._get(
            "/search/movie",
            params=params,
        )

        results = data.get("results", [])

        if not results and year:
            params.pop("year", None)

            data = self._get(
                "/search/movie",
                params=params,
            )

            results = data.get("results", [])

        if not results:
            return None

        return self._select_best_match(
            results,
            title,
            year,
        )

    def _select_best_match(
        self,
        results,
        title,
        year=None,
    ):
        normalized_title = str(title).strip().lower()

        for movie in results:
            result_title = (
                movie.get("title", "")
                .strip()
                .lower()
            )

            original_title = (
                movie.get("original_title", "")
                .strip()
                .lower()
            )

            release_date = movie.get("release_date", "")

            release_year = None

            if release_date:
                try:
                    release_year = int(
                        release_date[:4]
                    )
                except ValueError:
                    pass

            title_matches = (
                result_title == normalized_title
                or original_title == normalized_title
            )

            if (
                title_matches
                and year
                and release_year == int(year)
            ):
                return movie

        for movie in results:
            result_title = (
                movie.get("title", "")
                .strip()
                .lower()
            )

            original_title = (
                movie.get("original_title", "")
                .strip()
                .lower()
            )

            if (
                result_title == normalized_title
                or original_title == normalized_title
            ):
                return movie

        return results[0]

    # ==================================================
    # TMDB metadata calls
    # ==================================================

    def get_movie_details(self, tmdb_id):
        return self._get(
            f"/movie/{tmdb_id}"
        )

    def get_movie_credits(self, tmdb_id):
        return self._get(
            f"/movie/{tmdb_id}/credits"
        )

    def get_movie_keywords(self, tmdb_id):
        return self._get(
            f"/movie/{tmdb_id}/keywords"
        )

    # ==================================================
    # Combined metadata
    # ==================================================

    def get_movie_metadata(
        self,
        tmdb_id,
        use_cache=True,
    ):
        if use_cache:
            cached = self.get_cached_metadata(
                tmdb_id
            )

            if cached is not None:
                cached["cache_hit"] = True
                return cached

        details = self.get_movie_details(
            tmdb_id
        )

        credits = self.get_movie_credits(
            tmdb_id
        )

        keyword_data = self.get_movie_keywords(
            tmdb_id
        )

        genres = [
            genre["name"]
            for genre in details.get(
                "genres",
                []
            )
        ]

        directors = [
            person["name"]
            for person in credits.get(
                "crew",
                []
            )
            if person.get("job") == "Director"
        ]

        director = (
            directors[0]
            if directors
            else None
        )

        cast = [
            person["name"]
            for person in credits.get(
                "cast",
                []
            )[:5]
        ]

        keywords = [
            keyword["name"]
            for keyword in keyword_data.get(
                "keywords",
                []
            )
        ]

        poster_path = details.get(
            "poster_path"
        )

        poster_url = (
            f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            if poster_path
            else None
        )

        metadata = {
            "tmdb_id": tmdb_id,
            "tmdb_title": details.get(
                "title"
            ),
            "original_title": details.get(
                "original_title"
            ),
            "release_date": details.get(
                "release_date"
            ),
            "overview": details.get(
                "overview"
            ),
            "genres": genres,
            "director": director,
            "cast": cast,
            "keywords": keywords,
            "runtime": details.get(
                "runtime"
            ),
            "popularity": details.get(
                "popularity"
            ),
            "vote_average": details.get(
                "vote_average"
            ),
            "vote_count": details.get(
                "vote_count"
            ),
            "poster_url": poster_url,
            "cache_hit": False,
        }

        self.save_metadata_to_cache(
            metadata
        )

        return metadata

    # ==================================================
    # Search + metadata + caching
    # ==================================================

    def find_movie(
        self,
        title,
        year=None,
    ):
        cached_tmdb_id = self.get_cached_match(
            title,
            year,
        )

        if cached_tmdb_id is not None:
            metadata = self.get_movie_metadata(
                cached_tmdb_id,
                use_cache=True,
            )

            metadata["match_cache_hit"] = True

            return metadata

        result = self.search_movie(
            title,
            year,
        )

        if result is None:
            return None

        tmdb_id = result["id"]

        self.save_match_to_cache(
            title,
            year,
            tmdb_id,
        )

        metadata = self.get_movie_metadata(
            tmdb_id,
            use_cache=True,
        )

        metadata["match_cache_hit"] = False

        return metadata


# ======================================================
# DataFrame enrichment
# ======================================================

def enrich_user_movies(
    user_movies,
    client,
    delay=0.05,
    progress_callback=None,
):
    required_columns = {
        "Name",
        "Year",
    }

    missing = (
        required_columns
        - set(user_movies.columns)
    )

    if missing:
        raise ValueError(
            "user_movies is missing: "
            + ", ".join(sorted(missing))
        )

    enriched_rows = []

    total = len(user_movies)

    for _, row in user_movies.iterrows():
        title = row["Name"]
        year = row["Year"]

        if pd.isna(year):
            year = None
        else:
            year = int(year)

        try:
            metadata = client.find_movie(
                title=title,
                year=year,
            )

        except requests.RequestException as exc:
            print(
                f"TMDB error for "
                f"{title} ({year}): {exc}"
            )

            metadata = None

        enriched_row = row.to_dict()

        if metadata:
            enriched_row.update(
                metadata
            )

            enriched_row["tmdb_match"] = True

        else:
            enriched_row["tmdb_id"] = None
            enriched_row["tmdb_match"] = False
            enriched_row["cache_hit"] = False
            enriched_row["match_cache_hit"] = False

        enriched_rows.append(
            enriched_row
        )

        if progress_callback:
            progress_callback(
                len(enriched_rows) / total
            )

        # Only a tiny delay is needed.
        # Cached movies won't make API requests anyway.
        if delay:
            time.sleep(delay)

    return pd.DataFrame(
        enriched_rows
    )