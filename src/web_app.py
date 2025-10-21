from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix
import re
from difflib import get_close_matches
import json

app = Flask(__name__)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlflow/mlruns")
EXPERIMENT_NAME = "movie-recommender"

# -----------------------------
# Load movie titles
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
    
    movies["search_title"] = movies["title"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return movies

movies_df = load_movies()

# -----------------------------
# Movie matching function
# -----------------------------
def find_movie_id(movie_title, threshold=0.6):
    clean_input = re.sub(r'[^\w\s]', '', movie_title.lower().strip())
    
    exact_match = movies_df[movies_df["search_title"] == clean_input]
    if not exact_match.empty:
        return exact_match.iloc[0]["movieId"], exact_match.iloc[0]["title"]
    
    partial_matches = movies_df[movies_df["search_title"].str.contains(clean_input, na=False)]
    if not partial_matches.empty:
        return partial_matches.iloc[0]["movieId"], partial_matches.iloc[0]["title"]
    
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
        ratings = pd.read_csv(
            "ml-100k/u.data",
            sep="\t",
            names=["userId","movieId","rating","timestamp"]
        ).drop(columns=["timestamp"])
        
        self.user_item_matrix = ratings.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
    
    def get_recommendations(self, top_k_users=2, top_n_movies=5):
        if not self.user_ratings:
            return None, None
            
        sample_input = pd.DataFrame(0, index=[0], columns=self.user_item_matrix.columns)
        for mid, rating in self.user_ratings.items():
            if mid in sample_input.columns:
                sample_input.at[0, mid] = rating
        
        sparse_input = csr_matrix(sample_input.values)
        distances, indices = self.model.kneighbors(sparse_input, n_neighbors=top_k_users)
        
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()]
        mean_ratings = neighbor_ratings.mean(axis=0)
        
        unseen_movies = sample_input.columns[sample_input.iloc[0] == 0]
        recommended = mean_ratings[unseen_movies].sort_values(ascending=False).head(top_n_movies)
        
        return recommended.index.tolist(), recommended.values.tolist()
    
    def process_message(self, message, session_id):
        message = message.strip().lower()
        
        if session_id not in self.user_ratings:
            self.user_ratings[session_id] = {}
        
        user_session_ratings = self.user_ratings[session_id]
        
        if message in ['help', '?', 'commands']:
            return self.show_help()
        
        elif message in ['ratings', 'my ratings', 'show ratings']:
            return self.show_ratings(user_session_ratings)
        
        elif message in ['recommend', 'recommendations', 'suggest']:
            return self.generate_recommendations(user_session_ratings)
        
        elif message in ['clear', 'reset', 'start over']:
            return self.clear_ratings(session_id)
        
        else:
            return self.add_rating(message, user_session_ratings, session_id)
    
    def show_help(self):
        return {
            "type": "help",
            "message": """
🤖 <strong>Movie Recommendation Bot Commands:</strong>

🎬 <strong>Add Ratings:</strong>
   - <code>Movie Name:Rating</code> (e.g., <code>Toy Story:5</code>, <code>The Godfather:4</code>)

📋 <strong>View & Manage:</strong>
   - <code>ratings</code> - Show your current ratings
   - <code>clear</code> - Clear all ratings and start over
   - <code>recommend</code> - Get movie recommendations

❓ <strong>Help:</strong>
   - <code>help</code> - Show this help message
"""
        }
    
    def show_ratings(self, user_ratings):
        if not user_ratings:
            return {
                "type": "message",
                "message": "You haven't rated any movies yet. Add some ratings with `Movie:Rating`!"
            }
        
        ratings_text = "📊 <strong>Your Current Ratings:</strong><br>"
        for movie_id, rating in user_ratings.items():
            title = movies_df[movies_df.movieId == movie_id]['title'].values[0]
            ratings_text += f"⭐ {title} - {rating}/5<br>"
        
        return {
            "type": "ratings",
            "message": ratings_text,
            "count": len(user_ratings)
        }
    
    def add_rating(self, message, user_ratings, session_id):
        try:
            if ":" not in message:
                return {
                    "type": "error",
                    "message": "❌ Please use format: <code>Movie Name:Rating</code> (e.g., <code>Toy Story:5</code>)"
                }
            
            title, rating_str = message.rsplit(":", 1)
            title = title.strip()
            
            try:
                rating = float(rating_str.strip())
                if rating < 1 or rating > 5:
                    return {
                        "type": "error", 
                        "message": "❌ Rating must be between 1 and 5"
                    }
            except ValueError:
                return {
                    "type": "error",
                    "message": "❌ Rating must be a number between 1 and 5"
                }
            
            movie_id, matched_title = find_movie_id(title)
            
            if movie_id is not None:
                user_ratings[movie_id] = rating
                return {
                    "type": "rating_added",
                    "message": f"✅ Rated <strong>{matched_title}</strong> - {rating}/5",
                    "movie": matched_title,
                    "rating": rating
                }
            else:
                similar = self.find_similar_movies(title)
                if similar:
                    suggestions_text = f"❌ Movie '{title}' not found. Did you mean:<br>"
                    for suggestion in similar[:3]:
                        suggestions_text += f"• {suggestion}<br>"
                    return {
                        "type": "suggestions",
                        "message": suggestions_text,
                        "suggestions": similar[:3]
                    }
                else:
                    return {
                        "type": "error",
                        "message": f"❌ Movie '{title}' not found. Try being more specific or check spelling."
                    }
                
        except Exception as e:
            return {
                "type": "error",
                "message": f"❌ Error processing your rating: {str(e)}"
            }
    
    def find_similar_movies(self, title):
        clean_input = re.sub(r'[^\w\s]', '', title.lower().strip())
        all_titles = movies_df["title"].tolist()
        matches = get_close_matches(clean_input, [t.lower() for t in all_titles], n=3, cutoff=0.3)
        
        original_titles = []
        for match in matches:
            original = [t for t in all_titles if t.lower() == match]
            if original:
                original_titles.append(original[0])
        
        return original_titles
    
    def generate_recommendations(self, user_ratings):
        if len(user_ratings) < 1:
            return {
                "type": "error",
                "message": "❌ Please rate at least 1 movie before getting recommendations. Use <code>Movie:Rating</code>"
            }
        
        sample_input = pd.DataFrame(0, index=[0], columns=self.user_item_matrix.columns)
        for mid, rating in user_ratings.items():
            if mid in sample_input.columns:
                sample_input.at[0, mid] = rating
        
        movie_ids, scores = self.get_recommendations_from_vector(sample_input)
        
        if not movie_ids:
            return {
                "type": "error", 
                "message": "❌ Could not generate recommendations. Try rating more movies."
            }
        
        recommendations = []
        for mid, score in zip(movie_ids, scores):
            title = movies_df[movies_df.movieId == mid]['title'].values[0]
            recommendations.append({"title": title, "score": round(score, 2)})
        
        recommendations_text = f"🎯 <strong>Recommended Movies For You:</strong> (based on {len(user_ratings)} ratings)<br><br>"
        for rec in recommendations:
            recommendations_text += f"🎥 <strong>{rec['title']}</strong> (score: {rec['score']})<br>"
        
        return {
            "type": "recommendations",
            "message": recommendations_text,
            "recommendations": recommendations
        }
    
    def get_recommendations_from_vector(self, sample_input):
        sparse_input = csr_matrix(sample_input.values)
        distances, indices = self.model.kneighbors(sparse_input, n_neighbors=2)
        
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()]
        mean_ratings = neighbor_ratings.mean(axis=0)
        
        unseen_movies = sample_input.columns[sample_input.iloc[0] == 0]
        recommended = mean_ratings[unseen_movies].sort_values(ascending=False).head(5)
        
        return recommended.index.tolist(), recommended.values.tolist()
    
    def clear_ratings(self, session_id):
        count = len(self.user_ratings.get(session_id, {}))
        self.user_ratings[session_id] = {}
        return {
            "type": "message",
            "message": f"✅ Cleared all {count} ratings. Ready to start fresh!"
        }

bot = MovieRecommendationBot()

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie Recommendation System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            header {
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }
            h1 { font-size: 2.5em; margin-bottom: 10px; }
            .subtitle { opacity: 0.9; margin-bottom: 20px; }
            .chat-container {
                height: 500px;
                display: flex;
                flex-direction: column;
            }
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            .message { margin-bottom: 15px; display: flex; }
            .user-message { justify-content: flex-end; }
            .bot-message { justify-content: flex-start; }
            .message-content {
                max-width: 70%;
                padding: 12px 18px;
                border-radius: 18px;
                line-height: 1.4;
            }
            .user-message .message-content {
                background: #007bff;
                color: white;
                border-bottom-right-radius: 5px;
            }
            .bot-message .message-content {
                background: white;
                color: #333;
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .input-container {
                display: flex;
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }
            #messageInput {
                flex: 1;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
            }
            #messageInput:focus {
                border-color: #007bff;
            }
            #sendButton {
                background: #007bff;
                color: white;
                border: none;
                padding: 12px 25px;
                margin-left: 10px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
            }
            #sendButton:hover {
                background: #0056b3;
            }
            .quick-actions {
                display: flex;
                gap: 10px;
                padding: 15px 20px;
                background: #f8f9fa;
                border-top: 1px solid #e0e0e0;
                flex-wrap: wrap;
            }
            .quick-btn {
                background: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 15px;
                cursor: pointer;
                font-size: 14px;
            }
            .quick-btn:hover {
                background: #545b62;
            }
            .typing {
                opacity: 0.7;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎬 Movie Recommendation Bot</h1>
                <p class="subtitle">Rate movies and get personalized recommendations!</p>
            </header>

            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot-message">
                        <div class="message-content">
                            <strong>🤖 MovieBot:</strong> Welcome! Rate movies using "Movie:Rating" format (e.g., "Toy Story:5"). 
                            Type 'help' for commands or 'recommend' when ready!
                        </div>
                    </div>
                </div>

                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message... (e.g., The Godfather:5)" autocomplete="off">
                    <button id="sendButton">Send</button>
                </div>

                <div class="quick-actions">
                    <button class="quick-btn" onclick="sendQuickCommand('help')">Help</button>
                    <button class="quick-btn" onclick="sendQuickCommand('ratings')">My Ratings</button>
                    <button class="quick-btn" onclick="sendQuickCommand('recommend')">Get Recommendations</button>
                    <button class="quick-btn" onclick="sendQuickCommand('clear')">Clear All</button>
                </div>
            </div>
        </div>

        <script>
            let sessionId = 'user_' + Math.random().toString(36).substr(2, 9);

            function addMessage(content, isUser = false) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = content;
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function showTyping() {
                const chatMessages = document.getElementById('chatMessages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot-message';
                typingDiv.id = 'typingIndicator';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content typing';
                contentDiv.textContent = '🤖 MovieBot is typing...';
                
                typingDiv.appendChild(contentDiv);
                chatMessages.appendChild(typingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function hideTyping() {
                const typingIndicator = document.getElementById('typingIndicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }

            async function sendMessage() {
                const messageInput = document.getElementById('messageInput');
                const message = messageInput.value.trim();
                
                if (!message) return;

                // Add user message
                addMessage(`<strong>You:</strong> ${message}`, true);
                messageInput.value = '';

                // Show typing indicator
                showTyping();

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });

                    const data = await response.json();
                    hideTyping();
                    
                    // Add bot response
                    addMessage(`<strong>🤖 MovieBot:</strong> ${data.message}`);

                } catch (error) {
                    hideTyping();
                    addMessage(`<strong>🤖 MovieBot:</strong> ❌ Error: Could not connect to server`);
                }
            }

            function sendQuickCommand(command) {
                const messageInput = document.getElementById('messageInput');
                messageInput.value = command;
                sendMessage();
            }

            // Event listeners
            document.getElementById('sendButton').addEventListener('click', sendMessage);
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // Focus input on load
            document.getElementById('messageInput').focus();
        </script>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        response = bot.process_message(message, session_id)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "type": "error",
            "message": f"❌ Server error: {str(e)}"
        }), 500

@app.route('/movies/search', methods=['GET'])
def search_movies():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify([])
    
    matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
    results = matches.head(10)['title'].tolist()
    
    return jsonify(results)

# Remove or comment out this existing block:
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

# Add this instead:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)