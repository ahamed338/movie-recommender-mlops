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
    return f"runs:/{run.info.run_id}/model"

# -----------------------------
# Recommendation Engine
# -----------------------------
class MovieRecommendationBot:
    def __init__(self):
        self.user_ratings = {}
        self.load_data()
        self.model_uri = get_latest_model_uri()
        self.model = mlflow.sklearn.load_model(self.model_uri)
        
    def load_data(self):
        """Load ratings and create user-item matrix"""
        ratings = pd.read_csv(
            "ml-100k/u.data",
            sep="\t",
            names=["userId","movieId","rating","timestamp"]
        ).drop(columns=["timestamp"])
        
        self.user_item_matrix = ratings.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
    
    def get_recommendations(self, top_k_users=2, top_n_movies=5):
        """Get recommendations based on current user ratings"""
        if not self.user_ratings:
            return None, None
            
        # Create user vector
        sample_input = pd.DataFrame(0, index=[0], columns=self.user_item_matrix.columns)
        for mid, rating in self.user_ratings.items():
            if mid in sample_input.columns:
                sample_input.at[0, mid] = rating
        
        # Get recommendations
        sparse_input = csr_matrix(sample_input.values)
        distances, indices = self.model.kneighbors(sparse_input, n_neighbors=top_k_users)
        
        # Aggregate ratings from nearest neighbors
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()]
        mean_ratings = neighbor_ratings.mean(axis=0)
        
        # Recommend movies the user hasn't rated yet
        unseen_movies = sample_input.columns[sample_input.iloc[0] == 0]
        recommended = mean_ratings[unseen_movies].sort_values(ascending=False).head(top_n_movies)
        
        return recommended.index.tolist(), recommended.values.tolist()
    
    def process_message(self, message):
        """Process user message and return bot response"""
        message = message.strip().lower()
        
        # Help command
        if message in ['help', '?', 'commands']:
            return self.show_help()
        
        # Show current ratings
        elif message in ['ratings', 'my ratings', 'show ratings']:
            return self.show_ratings()
        
        # Get recommendations
        elif message in ['recommend', 'recommendations', 'suggest', 'what should i watch']:
            return self.generate_recommendations()
        
        # Clear ratings
        elif message in ['clear', 'reset', 'start over']:
            return self.clear_ratings()
        
        # Exit
        elif message in ['quit', 'exit', 'bye', 'goodbye']:
            return "Goodbye! Thanks for using MovieBot 🎬"
        
        # Add rating
        else:
            return self.add_rating(message)
    
    def show_help(self):
        """Show available commands"""
        help_text = """
🤖 **Movie Recommendation Bot Commands:**

🎬 **Add Ratings:**
   - `Movie Name:Rating` (e.g., `Toy Story:5`, `The Godfather:4`)
   - `Movie Name (Year):Rating` (e.g., `Pulp Fiction (1994):5`)

📋 **View & Manage:**
   - `ratings` - Show your current ratings
   - `clear` - Clear all ratings and start over
   - `recommend` - Get movie recommendations

❓ **Help:**
   - `help` - Show this help message
   - `quit` - Exit the bot

**Examples:**
- `The Shawshank Redemption:5`
- `ratings`
- `recommend`
- `clear`
"""
        return help_text
    
    def show_ratings(self):
        """Show current user ratings"""
        if not self.user_ratings:
            return "You haven't rated any movies yet. Add some ratings with `Movie:Rating`!"
        
        ratings_text = "📊 **Your Current Ratings:**\n"
        for movie_id, rating in self.user_ratings.items():
            title = movies_df[movies_df.movieId == movie_id]['title'].values[0]
            ratings_text += f"⭐ {title} - {rating}/5\n"
        
        ratings_text += f"\nTotal: {len(self.user_ratings)} movies rated"
        return ratings_text
    
    def add_rating(self, message):
        """Process rating addition"""
        try:
            if ":" not in message:
                return "❌ Please use format: `Movie Name:Rating` (e.g., `Toy Story:5`)"
            
            title, rating_str = message.rsplit(":", 1)
            title = title.strip()
            
            try:
                rating = float(rating_str.strip())
                if rating < 1 or rating > 5:
                    return "❌ Rating must be between 1 and 5"
            except ValueError:
                return "❌ Rating must be a number between 1 and 5"
            
            # Find movie
            movie_id, matched_title = find_movie_id(title, movies_df)
            
            if movie_id is not None:
                self.user_ratings[movie_id] = rating
                return f"✅ Rated **{matched_title}** - {rating}/5"
            else:
                # Suggest similar movies
                similar = self.find_similar_movies(title)
                if similar:
                    return f"❌ Movie '{title}' not found. Did you mean:\n" + "\n".join([f"  - {movie}" for movie in similar[:3]])
                else:
                    return f"❌ Movie '{title}' not found. Try being more specific or check spelling."
                
        except Exception as e:
            return f"❌ Error processing your rating: {str(e)}"
    
    def find_similar_movies(self, title):
        """Find similar movie titles for suggestions"""
        clean_input = re.sub(r'[^\w\s]', '', title.lower().strip())
        all_titles = movies_df["title"].tolist()
        matches = get_close_matches(clean_input, [t.lower() for t in all_titles], n=3, cutoff=0.3)
        
        # Get original titles back
        original_titles = []
        for match in matches:
            original = [t for t in all_titles if t.lower() == match]
            if original:
                original_titles.append(original[0])
        
        return original_titles
    
    def generate_recommendations(self):
        """Generate and return recommendations"""
        if len(self.user_ratings) < 1:
            return "❌ Please rate at least 1 movie before getting recommendations. Use `Movie:Rating`"
        
        movie_ids, scores = self.get_recommendations()
        
        if not movie_ids:
            return "❌ Could not generate recommendations. Try rating more movies."
        
        recommendations_text = "🎯 **Recommended Movies For You:**\n"
        for mid, score in zip(movie_ids, scores):
            title = movies_df[movies_df.movieId == mid]['title'].values[0]
            recommendations_text += f"🎥 **{title}** (score: {score:.2f})\n"
        
        recommendations_text += f"\nBased on your {len(self.user_ratings)} ratings. Add more ratings for better recommendations!"
        return recommendations_text
    
    def clear_ratings(self):
        """Clear all user ratings"""
        count = len(self.user_ratings)
        self.user_ratings = {}
        return f"✅ Cleared all {count} ratings. Ready to start fresh!"

# -----------------------------
# Chat Interface
# -----------------------------
def run_chatbot():
    bot = MovieRecommendationBot()
    
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION CHATBOT")
    print("=" * 60)
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            response = bot.process_message(user_input)
            print(f"\nBot: {response}\n")
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using MovieBot! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

if __name__ == "__main__":
    run_chatbot()