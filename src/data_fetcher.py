# src/data_fetcher.py
import requests
import pandas as pd
import json
import os
import time
from config.tmdb_config import TMDB_API_KEY, TMDB_BASE_URL

def fetch_popular_hindi_movies():
    """Fetch popular Hindi movies from TMDB API - ADDITION to English movies"""
    hindi_movies = []
    
    print("🔄 Fetching Hindi movies from TMDB API...")
    
    # Fetch multiple pages to get more movies
    for page in range(1, 4):  # Get ~60 movies
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/discover/movie",
                params={
                    'api_key': TMDB_API_KEY,
                    'with_original_language': 'hi',
                    'sort_by': 'popularity.desc',
                    'page': page,
                    'primary_release_date.gte': '1990-01-01',  # Movies from 1990 onwards
                },
                timeout=10
            )
            
            if response.status_code == 200:
                movies = response.json()['results']
                for movie in movies:
                    if movie.get('release_date'):
                        year = movie['release_date'][:4]
                        hindi_movies.append({
                            'movieId': movie['id'] + 10000,  # Offset to avoid ID conflicts with English movies
                            'title': f"{movie['title']} ({year})",
                            'original_title': movie.get('original_title', ''),
                            'popularity': movie['popularity'],
                            'vote_average': movie['vote_average'],
                            'vote_count': movie['vote_count'],
                            'release_year': year,
                            'language': 'hindi'
                        })
                print(f"✅ Page {page}: Found {len(movies)} Hindi movies")
            else:
                print(f"❌ API Error on page {page}: {response.status_code}")
                
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}")
            continue
    
    # Save to cache
    os.makedirs('data', exist_ok=True)
    with open('data/hindi_movies.json', 'w') as f:
        json.dump(hindi_movies, f, indent=2)
    
    print(f"✅ Total fetched: {len(hindi_movies)} Hindi movies")
    return pd.DataFrame(hindi_movies)

def load_hindi_movies():
    """Load Hindi movies from cache or fetch from API"""
    cache_file = 'data/hindi_movies.json'
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                movies_data = json.load(f)
            print(f"✅ Loaded {len(movies_data)} Hindi movies from cache")
            return pd.DataFrame(movies_data)
        except Exception as e:
            print(f"❌ Error loading cache: {e}. Fetching from API...")
            return fetch_popular_hindi_movies()
    else:
        return fetch_popular_hindi_movies()

# Test function
if __name__ == "__main__":
    movies = load_hindi_movies()
    print("\n🎬 Sample Hindi movies:")
    print(movies[['movieId', 'title', 'vote_average']].head(10))