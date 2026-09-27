from surprise import SVD, CoClustering, KNNBasic

### TUNE THE MODELS AND USE CORRECT PARAMS, FIND CORRECT WEIGHTS
def train_models(trainset):

    svd = SVD(
        n_factors=100,
        n_epochs=30,
        lr_all=0.01,
        reg_all=0.2
    )

    cocluster = CoClustering(
        n_cltr_u=10,
        n_cltr_i=10,
        n_epochs=10
    )

    svd.fit(trainset)
    cocluster.fit(trainset)

    models = {
        "svd": {
            "model": svd,
            "weight": 0.5
        },
        "cocluster": {
            "model": cocluster,
            "weight": 0.5
        },
    }

    return models


def ensemble_recommendations(
    models,
    userId,
    combined_ratings,
    movie_catalog,
    n_recs=10
):

    # Movies already rated by this user
    rated_movies = set(
        combined_ratings.loc[
            combined_ratings["userID"] == userId,
            "tmdbId"
        ]
    )

    # Movies eligible for recommendation
    candidates = (
        movie_catalog[
            ["tmdbId", "title"]
        ]
        .drop_duplicates(subset="tmdbId")
        .copy()
    )

    # Don't recommend movies already rated
    candidates = candidates[
        ~candidates["tmdbId"].isin(rated_movies)
    ].copy()

    # Generate prediction from every model
    for name, info in models.items():

        candidates[name] = candidates["tmdbId"].apply(
            lambda movie_id: info["model"]
            .predict(userId, movie_id)
            .est
        )

    # Weighted ensemble score
    candidates["ensemble_score"] = sum(
        candidates[name] * info["weight"]
        for name, info in models.items()
    )

    # Highest-scoring movies
    return (
        candidates
        .sort_values(
            "ensemble_score",
            ascending=False
        )
        .head(n_recs)
        .reset_index(drop=True)
    )