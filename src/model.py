import pandas as pd
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
