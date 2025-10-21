import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix
import re
from difflib import get_close_matches

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlflow/mlruns")
EXPERIMENT_NAME = "movie-recommender"

# -----------------------------
# Load movie titles with better preprocessing
# -----------------------------
def load_movies():
    movies = pd.read_csv(
        "ml-100k/u.item",
        sep="|",
        names=[
            "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
            "unknown", "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
            "War", "Western"
        ],
        encoding="latin-1"
    )[["movieId", "title"]]
    
    # Create search-friendly titles
    movies["search_title"] = movies["title"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return movies

movies_df = load_movies()

# -----------------------------
# Improved movie title matching
# -----------------------------
def find_movie_id(movie_title, movies_df, threshold=0.6):
    """
    Find movie ID using fuzzy matching
    Returns: (movie_id, matched_title) or (None, None) if not found
    """
    # Clean input title
    clean_input = re.sub(r'[^\w\s]', '', movie_title.lower().strip())
    
    # Try exact match first
    exact_match = movies_df[movies_df["search_title"] == clean_input]
    if not exact_match.empty:
        return exact_match.iloc[0]["movieId"], exact_match.iloc[0]["title"]
    
    # Try partial match
    partial_matches = movies_df[movies_df["search_title"].str.contains(clean_input, na=False)]
    if not partial_matches.empty:
        return partial_matches.iloc[0]["movieId"], partial_matches.iloc[0]["title"]
    
    # Try fuzzy matching as last resort
    all_titles = movies_df["search_title"].tolist()
    matches = get_close_matches(clean_input, all_titles, n=1, cutoff=threshold)
    if matches:
        matched_row = movies_df[movies_df["search_title"] == matches[0]].iloc[0]
        return matched_row["movieId"], matched_row["title"]
    
    return None, None

# -----------------------------
# MLflow utility (unchanged)
# -----------------------------
def get_latest_model_uri():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise Exception("No runs found in MLflow experiment.")
    
    run = runs[0]
    print(f"Latest run ID: {run.info.run_id}")
    return f"runs:/{run.info.run_id}/model"

# -----------------------------
# Recommend function (unchanged)
# -----------------------------
def recommend_movies(user_vector, user_item_matrix, top_k_users=2, top_n_movies=5):
    """Return top-N recommended movies for a given user vector."""
    model_uri = get_latest_model_uri()
    print(f"Loading model from: {model_uri}")
    
    model = mlflow.sklearn.load_model(model_uri)
    sparse_input = csr_matrix(user_vector.values)
    
    distances, indices = model.kneighbors(sparse_input, n_neighbors=top_k_users)
    
    # Aggregate ratings from nearest neighbors
    neighbor_ratings = user_item_matrix.iloc[indices.flatten()]
    mean_ratings = neighbor_ratings.mean(axis=0)
    
    # Recommend movies the user hasn't rated yet
    unseen_movies = user_vector.columns[user_vector.iloc[0] == 0]
    recommended = mean_ratings[unseen_movies].sort_values(ascending=False).head(top_n_movies)
    
    return recommended.index.tolist(), recommended.values.tolist()

# -----------------------------
# Improved Main Function
# -----------------------------
if __name__ == "__main__":
    # Load full user-item matrix
    ratings = pd.read_csv(
        "ml-100k/u.data",
        sep="\t",
        names=["userId","movieId","rating","timestamp"]
    ).drop(columns=["timestamp"])
    
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Interactive input with better guidance
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    print("Enter your ratings for movies (1-5). Leave blank for movies you haven't rated.")
    print("Format examples:")
    print("  - Toy Story:5")
    print("  - Toy Story (1995):5")
    print("  - toy story:5")
    print("  - Multiple: Toy Story:5, Jumanji:4, Lion King:3")
    print()
    
    input_str = input("Your ratings: ")
    user_ratings = {}
    found_movies = []
    not_found_movies = []
    
    if input_str.strip():
        for pair in input_str.split(","):
            try:
                title, rating = pair.rsplit(":", 1)
                title = title.strip()
                rating = float(rating.strip())
                
                # Find movie using improved matching
                movie_id, matched_title = find_movie_id(title, movies_df)
                
                if movie_id is not None:
                    user_ratings[movie_id] = rating
                    found_movies.append(f"✓ {matched_title} -> {rating}")
                else:
                    not_found_movies.append(f"✗ {title}")
                    
            except ValueError:
                print(f"⚠️  Skipping invalid format: '{pair}' - use 'Movie:Rating'")
            except Exception as e:
                print(f"⚠️  Error processing '{pair}': {e}")

    # Show matching results
    if found_movies:
        print("\n✅ Matched movies:")
        for msg in found_movies:
            print(f"  {msg}")
    
    if not_found_movies:
        print("\n❌ Not found (check spelling):")
        for msg in not_found_movies:
            print(f"  {msg}")

    # Create new user vector
    sample_input = pd.DataFrame(0, index=[0], columns=user_item_matrix.columns)
    for mid, rating in user_ratings.items():
        if mid in sample_input.columns:
            sample_input.at[0, mid] = rating

    # Get recommendations if we have at least one rating
    if user_ratings:
        movie_ids, scores = recommend_movies(sample_input, user_item_matrix)

        print("\n🎯 Recommended movies for you:")
        for mid, score in zip(movie_ids, scores):
            title = movies_df[movies_df.movieId == mid]['title'].values[0]
            print(f"  🎥 {title} (score: {score:.2f})")
    else:
        print("\n❌ No valid ratings provided. Cannot generate recommendations.")