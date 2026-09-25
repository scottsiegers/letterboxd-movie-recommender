
LETTERBOXD MOVIE RECOMMENDER

This application takes in an individual's letterboxd data, matches it to TMDB data to get movie features, and Movie Lense review data.

PURPOSE
    Letterboxd currently does not have an algorithmic recommendation feature to suggest relevant movies to its users. Other media services 
    such as Spotify and Netflix have award winning recommendation systems and do not need to be improved upon. The lack of this feature
    within letterboxd presents a unique opportunity to develop one for a very popular application with many users and many ratings / reviews.
    I also am an avid user of Letterboxd (as are many of my friends), so this is an especially interesting project for me. 
    There is a case to be made that Letterboxd not having an algorithmic feed / recommendation feature is refreshing to its users as the 
    point of the app is to log your own ratings and see what your friends are watching / liking. Seeing other movies your friends like or 
    don't like acts as its own sort of recommendation system, just a social not algorithmic. This is why this is simply a project for my
    friends and I to use out of curiosity or help us pick movies!

Movie Recommendation Engine

Overview
Architecture
Data Sources
    Letterboxd user exports
    MovieLens 32M
    TMDB

Recommendation Approach
    SVD
    User-user collaborative filtering
    CoClustering
    Ensemble
    Future content-based component

Model Evaluation
    RMSE
    Precision@K
    Recall@K
    NDCG@K

Application
    Letterboxd CSV upload
    TMDB matching
    Personalized recommendations
    Streamlit UI

Project Structure
Setup / Installation
Screenshots





Can build a supervised learning model from just the letterboxd and TMDB movie catalog data
    Target: Movie Rating
    Known rating --> learned with trained letterboxd data
    Unknown rating --> predicted with model trained on letterboxd data (is there enough?)
    Recommend movies with high predicted rating
        Letterboxd ratings out of 5, others may be out of 10
        Movie Lens uses 0.5 to 5 star ratings, the same as Letterboxd


Can then build a more complex recommendation system with collaborative filtering and matrix factorization
    This needs the Movie Lense data of user reviews / ratings of movies
        Can perform TF-IDF evaluation with movie reviews
        Can create movie clusters based on reviews

But MovieLens data only goes through 2023!
    This introduces a cold start scenario to my recommendation system. Adding in items that no users have reviewed yet. This is 
    a good reason to take a hybrid approach to the recommendation system
    
Models to compare:
    Global popularity model
    KNNBasic
        user user
        item item
    SVD
    CoClustering