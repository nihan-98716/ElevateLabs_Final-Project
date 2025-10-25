# Movie Recommendation System
# This script builds a content-based movie recommendation system using the MovieLens dataset.
# It suggests movies to users based on their genre preferences.

# --- 1. Import Libraries ---
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import urllib.error

# --- 2. Load and Prepare the Data ---
# The MovieLens 100K dataset is a popular dataset for recommendation systems.
# We will download the data directly from the source URL to run in environments like Google Colab.

# URLs for the MovieLens 100K dataset
item_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
data_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'


try:
    # Load the movie data (u.item) from the URL
    # The columns are: movie_id, movie_title, release_date, ..., genres...
    # We specify the separator, encoding, and column names for clarity.
    movie_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                     'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(item_url, sep='|', names=movie_columns, encoding='latin-1')

    # Load the ratings data (u.data) from the URL
    # The columns are: user_id, movie_id, rating, timestamp
    rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_url, sep='\t', names=rating_columns)
except urllib.error.URLError as e:
    print("="*50)
    print(f"ERROR: Could not download the dataset files. Please check your internet connection.")
    print(f"Details: {e}")
    print("="*50)
    exit()
except Exception as e:
    print("="*50)
    print(f"An error occurred while processing the data: {e}")
    print("="*50)
    exit()


# --- 3. Data Preprocessing and Cleaning ---

# For our content-based filter, we only need movie information.
# Let's create a copy to work with.
movies_df = movies.copy()

# The genre information is spread across multiple columns (0s and 1s).
# We'll create a single 'genres' string for each movie.
# For example: "Action|Adventure|Sci-Fi"
genre_columns = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def get_genres(row):
    """Combines genre columns into a single space-separated string."""
    genres = []
    for genre in genre_columns:
        if row[genre] == 1:
            genres.append(genre)
    return ' '.join(genres)

movies_df['genres'] = movies_df.apply(get_genres, axis=1)

# We now have a clean dataframe with 'movie_id', 'movie_title', and 'genres'.
# Let's keep only the columns we need.
movies_df = movies_df[['movie_id', 'movie_title', 'genres']]


# --- 4. Building the Content-Based Filtering Model ---

# We will use TF-IDF (Term Frequency-Inverse Document Frequency) to convert our genre strings
# into a matrix of numerical features. This allows us to calculate similarity between movies.

# Initialize a TfidfVectorizer.
# 'stop_words='english'' removes common words that don't add much meaning.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Replace any missing genre values with an empty string
movies_df['genres'] = movies_df['genres'].fillna('')

# Fit and transform the data to create the TF-IDF matrix.
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])


# Now, we compute the cosine similarity matrix.
# Cosine similarity measures the cosine of the angle between two vectors,
# giving us a similarity score between 0 and 1.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# We need a way to map movie titles to their index in the dataframe.
indices = pd.Series(movies_df.index, index=movies_df['movie_title']).drop_duplicates()


# --- 5. Creating the Recommendation Function ---

def get_recommendations_by_genre(genre, cosine_sim=cosine_sim, movies_df=movies_df):
    """
    This function recommends movies based on a selected genre.
    It finds movies of that genre and then lists other similar movies.
    """
    # Find movies that belong to the specified genre
    genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]

    if genre_movies.empty:
        return f"No movies found for the genre: {genre}"

    # We will use the most popular movie in that genre as our "seed" movie.
    # To find the most popular, we can look at the ratings data.
    movie_ratings = ratings.groupby('movie_id').size().reset_index(name='num_ratings')
    genre_movies_with_ratings = pd.merge(genre_movies, movie_ratings, on='movie_id')

    if genre_movies_with_ratings.empty:
        # If no ratings, just pick the first movie of that genre
        seed_movie_title = genre_movies.iloc[0]['movie_title']
    else:
        # Get the title of the most rated movie in that genre
        seed_movie_title = genre_movies_with_ratings.sort_values('num_ratings', ascending=False).iloc[0]['movie_title']


    # Get the index of our seed movie
    try:
        idx = indices[seed_movie_title]
    except KeyError:
        return f"Could not find index for movie: {seed_movie_title}"


    # Get the pairwise similarity scores of all movies with our seed movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (we skip the first one, as it's the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return movies_df['movie_title'].iloc[movie_indices]


# --- 6. User Interaction and Displaying Results ---

if __name__ == '__main__':
    print("*" * 50)
    print("Movie Recommendation System")
    print("*" * 50)

    # List available genres for the user to choose from
    print("\nAvailable genres:")
    for g in genre_columns:
        print(f"- {g}")

    # Get user input
    selected_genre = input("\nEnter a genre to get recommendations: ")

    # Validate user input
    if selected_genre.capitalize() not in genre_columns and selected_genre not in genre_columns:
         # A simple check to see if the input is a valid genre
         if selected_genre in movies_df['genres'].to_string():
             pass # allow partial matches like 'sci' for 'Sci-Fi'
         else:
            print(f"\n'{selected_genre}' is not a valid genre. Please choose from the list above.")
            exit()


    # Get and display recommendations
    recommendations = get_recommendations_by_genre(selected_genre)

    print("\n-------------------------------------------------")
    print(f"Top 10 recommendations for '{selected_genre}':")
    print("-------------------------------------------------")
    if isinstance(recommendations, pd.Series):
        for i, movie in enumerate(recommendations):
            print(f"{i+1}. {movie}")
    else:
        print(recommendations)
    print("\n")
