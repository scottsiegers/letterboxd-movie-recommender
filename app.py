import streamlit as st
import pandas as pd
from surprise import dump
from src.recommender.model import (train_models, ensemble_recommendations)
from surprise import Dataset, Reader

from src.letterboxd_parser import (
    parse_letterboxd_uploads,
    build_user_movies,
    get_upload_summary,
    get_detected_files,
)
from src.tmdb import TMDBClient, enrich_user_movies


# LOCATION: app.py

@st.cache_data
def load_base_ratings():
    return pd.read_parquet(
        "data/combined_ratings_final.parquet"
    )


base_ratings = load_base_ratings()

@st.cache_data
def load_movie_catalog():
    return pd.read_parquet(
        "data/movie_catalog.parquet"
    )

movie_catalog = load_movie_catalog()

@st.cache_resource
def load_models():

    _, svd = dump.load("models/svd.pkl")
    _, cocluster = dump.load("models/cocluster.pkl")
    _, user_knn = dump.load("models/user_knn.pkl")

    models = {
        "svd": {
            "model": svd,
            "weight": 0.5
        },
        "cocluster": {
            "model": cocluster,
            "weight": 0.3
        },
        "user_knn": {
            "model": user_knn,
            "weight": 0.2
        }
    }

    return models

models = load_models()


st.set_page_config(
    page_title="Letterboxd Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Letterboxd Recommender")

st.write(
    "Upload your Letterboxd data, connect it to TMDB, "
    "and start building your personalized movie profile."
)

st.info(
    "You can upload a Letterboxd ZIP, select the export folder, "
    "or upload CSV files individually. ratings.csv is required."
)


# ==========================================================
# TMDB client
# ==========================================================

try:
    tmdb = TMDBClient()
    tmdb_connected = True

except Exception as e:
    tmdb_connected = False

    st.error(
        "TMDB is not connected. Make sure TMDB_TOKEN "
        "is available in your environment or .env file."
    )

    st.caption(str(e))


# ==========================================================
# Upload method
# ==========================================================

upload_method = st.radio(
    "How would you like to upload your Letterboxd data?",
    [
        "ZIP file",
        "Export folder",
        "Individual CSV files",
    ],
    horizontal=True,
)


zip_file = None
folder_files = None
csv_files = None


if upload_method == "ZIP file":

    zip_file = st.file_uploader(
        "Upload your Letterboxd export ZIP",
        type=["zip"],
        key="zip_upload",
    )


elif upload_method == "Export folder":

    folder_files = st.file_uploader(
        "Select your Letterboxd export folder",
        type=["csv"],
        accept_multiple_files="directory",
        key="folder_upload",
    )


elif upload_method == "Individual CSV files":

    csv_files = st.file_uploader(
        "Upload Letterboxd CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_upload",
    )

    st.caption(
        "You can upload ratings.csv, reviews.csv, diary.csv, "
        "watched.csv, and watchlist.csv together."
    )


has_upload = (
    zip_file is not None
    or bool(folder_files)
    or bool(csv_files)
)


# ==========================================================
# Process Letterboxd upload
# ==========================================================

if has_upload:

    try:

        data = parse_letterboxd_uploads(
            zip_file=zip_file,
            folder_files=folder_files,
            csv_files=csv_files,
        )

        detected_files = get_detected_files(data)

        if not detected_files:
            st.warning(
                "No supported Letterboxd files were found."
            )
            st.stop()

        st.subheader("Import Summary")

        st.write(
            "Detected: "
            + ", ".join(
                f"{name}.csv"
                for name in detected_files
            )
        )

        summary = get_upload_summary(data)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Ratings",
                summary["ratings"],
            )

        with col2:
            st.metric(
                "Reviews",
                summary["reviews"],
            )

        with col3:
            st.metric(
                "Watched",
                summary["watched"],
            )

        with col4:
            st.metric(
                "Diary",
                summary["diary"],
            )

        with col5:
            st.metric(
                "Watchlist",
                summary["watchlist"],
            )


        # ==================================================
        # Build user movie dataframe
        # ==================================================

        user_movies = build_user_movies(data)

        st.success(
            f"Successfully loaded "
            f"{len(user_movies):,} rated movies."
        )


        # ==================================================
        # Letterboxd statistics
        # ==================================================

        st.subheader("Your Letterboxd Stats")

        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.metric(
                "Movies Rated",
                len(user_movies),
            )

        with stat2:
            avg_rating = user_movies["Rating"].mean()

            st.metric(
                "Average Rating",
                f"{avg_rating:.2f} ★",
            )

        with stat3:

            if "Review" in user_movies.columns:
                review_count = (
                    user_movies["Review"]
                    .notna()
                    .sum()
                )

            else:
                review_count = 0

            st.metric(
                "Reviews Written",
                review_count,
            )


        # ==================================================
        # Highest-rated movies
        # ==================================================

        st.subheader("Your Highest Rated Movies")

        favorites = (
            user_movies
            .sort_values(
                ["Rating", "Year"],
                ascending=[False, False],
            )
            .head(20)
        )

        display_cols = [
            col
            for col in [
                "Name",
                "Year",
                "Rating",
                "Review",
            ]
            if col in favorites.columns
        ]

        st.dataframe(
            favorites[display_cols],
            use_container_width=True,
            hide_index=True,
        )


        # ==================================================
        # Rating distribution
        # ==================================================

        st.subheader("Your Rating Distribution")

        rating_counts = (
            user_movies["Rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_counts
        )


        # ==================================================
        # TMDB enrichment
        # ==================================================

        st.divider()

        st.subheader("TMDB Movie Matching")

        st.write(
            "Match your Letterboxd movies with TMDB to add "
            "genres, director, cast, plot summaries, keywords, "
            "posters, and stable movie IDs."
        )

        enrichment_limit = st.selectbox(
            "How many movies would you like to enrich?",
            [
                10,
                25,
                50,
                100,
                "All movies",
            ],
            index=0,
        )

        enrich_button = st.button(
            "Match Movies with TMDB",
            type="primary",
            disabled=not tmdb_connected,
        )


        # ==================================================
        # Run enrichment
        # ==================================================

        if enrich_button:

            if enrichment_limit == "All movies":
                movies_to_enrich = user_movies.copy()

            else:
                movies_to_enrich = user_movies.head(
                    enrichment_limit
                ).copy()


            progress_bar = st.progress(0)

            progress_text = st.empty()

            def update_progress(progress):

                progress_bar.progress(
                    min(progress, 1.0)
                )

                percent = int(
                    progress * 100
                )

                progress_text.write(
                    f"Matching movies with TMDB... "
                    f"{percent}%"
                )


            with st.spinner(
                "Retrieving movie metadata from TMDB..."
            ):

                enriched_movies = enrich_user_movies(
                    movies_to_enrich,
                    client=tmdb,
                    progress_callback=update_progress,
                )


            progress_bar.progress(1.0)

            progress_text.write(
                "TMDB matching complete."
            )


            # ==============================================
            # Store results in Streamlit session
            # ==============================================

            st.session_state[
                "enriched_movies"
            ] = enriched_movies


        # ==================================================
        # Display TMDB results
        # ==================================================

        if "enriched_movies" in st.session_state:

            enriched_movies = st.session_state[
                "enriched_movies"
            ]

            matched_count = int(
                enriched_movies[
                    "tmdb_match"
                ].sum()
            )

        cache_hits = 0

        if "cache_hit" in enriched_movies.columns:
            cache_hits = int(
                enriched_movies[
                    "cache_hit"
                ].fillna(False)
                .sum()
            )

            total_count = len(
                enriched_movies
            )

            unmatched_count = (
                total_count
                - matched_count
            )

            match_rate = (
                matched_count
                / total_count
                * 100
                if total_count
                else 0
            )


            st.subheader(
                "TMDB Match Results"
            )

            match1, match2, match3, match4 = (
                st.columns(4)
            )

            with match1:
                st.metric(
                    "Movies Matched",
                    matched_count,
                )

            with match2:
                st.metric(
                    "Not Matched",
                    unmatched_count,
                )

            with match3:
                st.metric(
                    "Match Rate",
                    f"{match_rate:.1f}%",
                )
            with match4:
                st.metric(
                    "Cache Hits",
                    cache_hits,
    )


            # ==============================================
            # Metadata table
            # ==============================================

            st.subheader(
                "Enriched Movie Data"
            )

            metadata_cols = [
                col
                for col in [
                    "Name",
                    "Year",
                    "Rating",
                    "tmdb_id",
                    "tmdb_title",
                    "genres",
                    "director",
                    "cast",
                    "vote_average",
                    "tmdb_match",
                ]
                if col
                in enriched_movies.columns
            ]

            st.dataframe(
                enriched_movies[
                    metadata_cols
                ],
                use_container_width=True,
                hide_index=True,
            )


            # ==============================================
            # Movie cards
            # ==============================================

            st.subheader(
                "Movie Preview"
            )

            matched_movies = (
                enriched_movies[
                    enriched_movies[
                        "tmdb_match"
                    ] == True
                ]
                .head(12)
            )

            for start in range(
                0,
                len(matched_movies),
                4,
            ):

                row_movies = (
                    matched_movies
                    .iloc[
                        start:start + 4
                    ]
                )

                columns = st.columns(4)

                for column, (_, movie) in zip(
                    columns,
                    row_movies.iterrows(),
                ):

                    with column:

                        poster_url = (
                            movie.get(
                                "poster_url"
                            )
                        )

                        if poster_url:
                            st.image(
                                poster_url,
                                use_container_width=True,
                            )

                        st.markdown(
                            f"### {movie['Name']}"
                        )

                        year = movie.get(
                            "Year"
                        )

                        rating = movie.get(
                            "Rating"
                        )

                        if year:
                            st.caption(
                                f"{int(year)}"
                            )

                        if rating:
                            st.write(
                                f"Your rating: "
                                f"**{rating} ★**"
                            )

                        director = (
                            movie.get(
                                "director"
                            )
                        )

                        if director:
                            st.write(
                                f"**Director:** "
                                f"{director}"
                            )

                        genres = movie.get(
                            "genres"
                        )

                        if isinstance(
                            genres,
                            list,
                        ):

                            st.write(
                                "**Genres:** "
                                + ", ".join(
                                    genres
                                )
                            )

                        overview = (
                            movie.get(
                                "overview"
                            )
                        )

                        if overview:

                            if len(
                                overview
                            ) > 220:

                                overview = (
                                    overview[
                                        :220
                                    ]
                                    + "..."
                                )

                            st.write(
                                overview
                            )


            # ==============================================
            # Unmatched movies
            # ==============================================

            unmatched_movies = (
                enriched_movies[
                    enriched_movies[
                        "tmdb_match"
                    ] == False
                ]
            )

            if not unmatched_movies.empty:

                with st.expander(
                    "View movies TMDB could not match"
                ):

                    st.dataframe(
                        unmatched_movies[
                            [
                                "Name",
                                "Year",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


            # ==============================================
            # Download enriched CSV
            # ==============================================

            csv_data = (
                enriched_movies
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            st.download_button(
                label=(
                    "Download Enriched Movie Data"
                ),
                data=csv_data,
                file_name=(
                    "letterboxd_tmdb_enriched.csv"
                ),
                mime="text/csv",
            )


        # ==================================================
        # Raw data
        # ==================================================

        with st.expander(
            "View all imported Letterboxd movies"
        ):

            st.dataframe(
                user_movies,
                use_container_width=True,
                hide_index=True,
            )


    except Exception as e:

        st.error(
            "Something went wrong: "
            f"{e}"
        )

st.header("Recommended for You")

if st.button("Get Recommendations"):

    new_user_id = int(
        base_ratings["userId"].max() + 1
    )

    user_ratings = enriched_movies[
    ["tmdb_id", "Rating"]
    ].copy()

    user_ratings = user_ratings.rename(
        columns={
            "tmdb_id": "tmdbId",
            "Rating": "rating"
        }
    )

    user_ratings["userID"] = new_user_id

    user_ratings = user_ratings[
        ["userID", "tmdbId", "rating"]
    ]

    known_movies = set(
        base_ratings["tmdbId"].unique()
    )

    user_collab_ratings = user_ratings[
        user_ratings["tmdbId"].isin(known_movies)
    ].copy()

    st.write("Combining ratings")
    combined_ratings = pd.concat(
        [
            base_ratings[
                ["userId", "tmdbId", "rating"]
            ],
            user_collab_ratings
        ],
        ignore_index=True
    )

    st.write("Building trainset")
    reader = Reader(
        rating_scale=(0.5, 5.0)
    )

    data = Dataset.load_from_df(
        combined_ratings[
            ["userID", "tmdbId", "rating"]
        ],
        reader
    )

    trainset = data.build_full_trainset()

    st.write("training models")
    models = train_models(trainset)

    st.write("Models trained")
    recommendations = ensemble_recommendations(
        models=models,
        userId=new_user_id,
        combined_ratings=combined_ratings,
        movie_catalog=movie_catalog,
        n_recs=10
    )
    st.write("Recommendations Complete")
    st.session_state["recommendations"] = recommendations

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    for start in range(0, len(recommendations), 5):

        row = recommendations.iloc[start:start + 5]

        columns = st.columns(5)

        for column, (_, movie) in zip(columns, row.iterrows()):

            with column:

                # if pd.notna(movie["poster_url"]):
                #     st.image(
                #         movie["poster_url"],
                #         use_container_width=True
                #     )

                st.markdown(
                    f"**{movie['title']}**"
                )