
LETTERBOXD MOVIE RECOMMENDER

This application takes in an individual's letterboxd data, matches it to TMDB data to get movie features, folds it into MovieLens 
32M ratings dataset, and recommends films based using a hybrid SVD / Content based model approach.

Hosted here: https://letterboxd-movie-recommender.streamlit.app

INSTRUCTIONS
    1. Download your perosnal letterboxd rating data here: https://letterboxd.com/user/exportdata/
    2. Choose 'Individual CSV files' and upload the 'ratings.csv' and 'reviews.csv' files 
       from the downloaded letterboxd zip (or just upload the whole zip file)
    3. Review your letterboxd data!
    4. Enrich your movies with TMDB data
    5. Get your recommendations!

PURPOSE
    Letterboxd currently does not have an algorithmic recommendation feature to suggest relevant movies to its users. Other media services 
    such as Spotify and Netflix have award winning recommendation systems and do not need to be improved upon. The lack of this feature
    within letterboxd presents a unique opportunity to develop one for a very popular application with many users and many ratings / reviews.
    I also am an avid user of Letterboxd (as are many of my friends), so this is an especially interesting project for me. 
    There is a case to be made that Letterboxd not having an algorithmic feed / recommendation feature is refreshing to its users as the 
    point of the app is to log your own ratings and see what your friends are watching / liking. Seeing other movies your friends like or 
    don't like acts as its own sort of recommendation system, just a social not algorithmic. This is why this is simply a project for my
    friends and I to use out of curiosity or help us pick movies!


MOVIE RECOMMENDATION ENGINE 

Architecture
    .ipynb Notebooks
        Notebook	                Main responsibility
        01_letterboxd_explore	    Understand user-export structure and ratings
        02_rating_predictor         Further analysis and simple predictions
        03_feature_engineering	    Clean and prepare movie metadata
        04_popularity_recommender   Develop popularity recommender for user cold start problem
        05_content_based_model	    Fit TF-IDF and build user profiles
        06_recommender_testing	    Ad-hoc testing of collaborative models and rankings
        07_model_comparison	        Compare models using consistent evaluation

Data Sources
    Letterboxd user exports (provided by individual users)
    MovieLens 32M rating dataset
    TMDB movie data  
    Main datasets created during 03_feature_engineering.ipynb   
        ./data/movie_with_features.parquet : movie catalog from 32M ML data with tmdbID features (director, genre, keywords, poster url)
        ./data/movielens_ratings_w_tmdbId: 32 ML ratings data with userId, tmdbID, rating

Model Evaluation
    RMSE
    Precision@K
    Recall@K
    NDCG@K

Application Flow
    Letterboxd CSV upload
    Quick profile analysis
    TMDB matching
    Personalized recommendations



