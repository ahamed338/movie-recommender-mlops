# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from utils import create_hybrid_ratings

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlflow/mlruns")
mlflow.set_experiment("hybrid-movie-recommender")  # Updated experiment name

# -----------------------------
# Load HYBRID dataset (ENGLISH + HINDI)
# -----------------------------
print("🔄 Loading hybrid dataset (English + Hindi movies)...")
ratings = create_hybrid_ratings()  # CHANGED: Uses combined data

# -----------------------------
# Create user-item matrix
# -----------------------------
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)

print(f"📊 Training matrix: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} movies")

# -----------------------------
# Train KNN collaborative filtering model
# -----------------------------
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# -----------------------------
# Log model to MLflow
# -----------------------------
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    
    # Log some metrics
    mlflow.log_param("total_users", user_item_matrix.shape[0])
    mlflow.log_param("total_movies", user_item_matrix.shape[1])
    mlflow.log_param("dataset", "hybrid_english_hindi")
    
    print("✅ Hybrid training complete! English + Hindi movies")
    print(f"Run ID: {run.info.run_id}")
    print(f"Model can recommend from {user_item_matrix.shape[1]} movies")