import os

# Define folder structure
folders = [
    "data",
    "mlflow/mlruns",
    "src",
    "notebooks",
    ".devcontainer",
    ".github/workflows"
]

# Define files with starter code content
files = {
    "data/movielens_sample.csv": "userId,movieId,rating,timestamp\n1,1,4,964982703\n1,3,4,964981247\n2,1,5,964982224\n",
    
    "src/train.py": '''import mlflow
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
''',

    "src/model.py": '''import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np

class RecommenderModel:
    def __init__(self):
        self.model = TruncatedSVD(n_components=20, random_state=42)
        self.user_factors = None
        self.movie_factors = None
        self.user_index = None
        self.movie_index = None

    def train(self, data: pd.DataFrame):
        user_movie_matrix = data.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        self.user_index = user_movie_matrix.index
        self.movie_index = user_movie_matrix.columns
        matrix = user_movie_matrix.values
        self.model.fit(matrix)
        self.user_factors = self.model.transform(matrix)
        self.movie_factors = self.model.components_
        print("Model trained.")

    def recommend(self, user_id, top_k=5):
        if user_id not in self.user_index:
            return []
        user_idx = self.user_index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        scores = np.dot(user_vector, self.movie_factors)
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = [self.movie_index[i] for i in top_indices]
        return recommendations
''',

    "src/predict.py": '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import uvicorn

app = FastAPI()

class UserRequest(BaseModel):
    user_id: int

# Load model once at startup
model = mlflow.sklearn.load_model("mlflow/mlruns/model")

@app.post("/recommend")
def recommend_movies(req: UserRequest):
    try:
        recs = model.recommend(req.user_id)
        return {"user_id": req.user_id, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',

    "src/utils.py": '''import pandas as pd

def load_data(path):
    return pd.read_csv(path)
''',

    "Dockerfile": '''FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    "requirements.txt": '''fastapi
uvicorn
pandas
scikit-learn
mlflow
''',

    ".devcontainer/devcontainer.json": '''{
    "name": "MLops Dev Container",
    "image": "mcr.microsoft.com/vscode/devcontainers/python:3.9",
    "postCreateCommand": "pip install -r requirements.txt"
}
''',

    ".github/workflows/mlops-ci-cd.yml": '''name: MLOps CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  schedule:
    - cron: '0 0 * * *'  # every day at midnight

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Train Model
      run: |
        python src/train.py

    - name: Build Docker image
      run: docker build -t movie-recommender .

    - name: Run tests
      run: |
        # Add your test scripts here (if any)
        echo "No tests yet"
'''
,
    "README.md": '''# Movie Recommender MLOps Project

This project is an end-to-end MLOps demo that trains a movie recommendation model on MovieLens dataset,
deploys it as a FastAPI service, and automates CI/CD pipelines using GitHub Actions and Codespaces.

## Structure

- `data/` - Sample data files
- `src/` - Source code for model training, prediction API, and utilities
- `.github/workflows/` - GitHub Actions workflows for CI/CD
- `.devcontainer/` - Configuration for GitHub Codespaces development environment

## How to run

1. Open the project in GitHub Codespaces.
2. The dependencies auto-install.
3. Run `python src/train.py` to train the model.
4. Run `python src/predict.py` to start the API.
5. Use the `/recommend` endpoint to get movie recommendations.
'''
}

# Function to create folders and files
def create_project():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    print("Project scaffold created successfully.")

if __name__ == "__main__":
    create_project()
