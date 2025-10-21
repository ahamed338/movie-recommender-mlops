import mlflow
import pandas as pd
from model import RecommenderModel
from utils import load_data

def train():
    mlflow.set_experiment("movie-recommender")
    with mlflow.start_run():
        data = load_data("data/movielens_sample.csv")
        model = RecommenderModel()
        model.train(data)
        mlflow.log_metric("trained_users", len(data['userId'].unique()))
        mlflow.sklearn.log_model(model, "model")
        print("Training complete and model logged.")

if __name__ == "__main__":
    train()
