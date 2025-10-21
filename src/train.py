import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlflow/mlruns")
mlflow.set_experiment("movie-recommender")

# -----------------------------
# Load MovieLens 100K dataset
# -----------------------------
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
).drop(columns=["timestamp"])

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
# Create user-item matrix
# -----------------------------
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)

# -----------------------------
# Train KNN collaborative filtering model
# -----------------------------
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# -----------------------------
# Log model to MLflow
# -----------------------------
with mlflow.start_run() as run:
    # Log without input_example to avoid warnings
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    
    print("Training complete ✅ | Model logged")
    print(f"Run ID: {run.info.run_id}")
