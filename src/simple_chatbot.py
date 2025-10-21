import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix
import re

# ... (include all the same imports and helper functions as above)

class SimpleMovieBot:
    def __init__(self):
        self.user_ratings = {}
        self.load_data()
        self.model = mlflow.sklearn.load_model(get_latest_model_uri())
    
    def load_data(self):
        ratings = pd.read_csv("ml-100k/u.data", sep="\t", 
                            names=["userId","movieId","rating","timestamp"]).drop(columns=["timestamp"])
        self.user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    def chat(self):
        print("🎬 Hi! I'm your movie recommendation bot!")
        print("Tell me movies you like with ratings (1-5), like 'Toy Story:5'")
        print("Type 'recommend' when ready, or 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Bot: Thanks for chatting! 🍿")
                break
                
            elif user_input.lower() == 'recommend':
                if self.user_ratings:
                    self.get_and_show_recommendations()
                else:
                    print("Bot: Please rate some movies first! Like 'The Godfather:5'")
                    
            elif ':' in user_input:
                self.process_rating(user_input)
            else:
                print("Bot: Use 'Movie:Rating' to rate movies, or 'recommend' to get suggestions")

    def process_rating(self, input_str):
        try:
            title, rating_str = input_str.rsplit(":", 1)
            title, rating = title.strip(), float(rating_str.strip())
            
            movie_id, matched_title = find_movie_id(title, movies_df)
            if movie_id:
                self.user_ratings[movie_id] = rating
                print(f"Bot: ✅ Rated '{matched_title}' - {rating}/5")
            else:
                print(f"Bot: ❌ '{title}' not found. Try another title!")
                
        except ValueError:
            print("Bot: ❌ Use format: Movie:Rating (e.g., Toy Story:5)")

    def get_and_show_recommendations(self):
        # ... (same recommendation logic as above)
        print("Bot: 🎯 Here are your recommendations...")
        # Implementation here

if __name__ == "__main__":
    bot = SimpleMovieBot()
    bot.chat()