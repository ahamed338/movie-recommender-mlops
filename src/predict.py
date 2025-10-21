import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlflow/mlruns")
EXPERIMENT_NAME = "movie-recommender"

# -----------------------------
# Load movie titles
# -----------------------------
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

# -----------------------------
# MLflow utility
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
# Recommend function
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
# Main
# -----------------------------
if __name__ == "__main__":
    # Load full user-item matrix
    ratings = pd.read_csv(
        "ml-100k/u.data",
        sep="\t",
        names=["userId","movieId","rating","timestamp"]
    ).drop(columns=["timestamp"])
    
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Interactive input
    print("Enter your ratings for movies (1-5). Leave blank for movies you haven't rated.")
    print("Format: Movie Name (Year):Rating, e.g., Toy Story (1995):5, Jumanji (1995):4")
    
    input_str = input("Your ratings: ")
    user_ratings = {}
    if input_str.strip():
        for pair in input_str.split(","):
            try:
                title, rating = pair.rsplit(":", 1)
                title = title.strip()
                rating = float(rating.strip())
                # Convert movie name to movieId
                movie_row = movies[movies.title.str.lower() == title.lower()]
                if not movie_row.empty:
                    movie_id = movie_row.iloc[0].movieId
                    user_ratings[movie_id] = rating
                else:
                    print(f"Movie not found: {title}")
            except:
                print(f"Skipping invalid input: {pair}")

    # Create new user vector aligned with training columns
    sample_input = pd.DataFrame(0, index=[0], columns=user_item_matrix.columns)
    for mid, rating in user_ratings.items():
        if mid in sample_input.columns:
            sample_input.at[0, mid] = rating

    # Get recommendations
    movie_ids, scores = recommend_movies(sample_input, user_item_matrix)

    print("\n✅ Recommended movies:")
    for mid, score in zip(movie_ids, scores):
        title = movies[movies.movieId == mid]['title'].values[0]
        print(f"{title} (score: {score:.2f})")
