# src/test_hindi.py
from utils import load_combined_movies

movies_df = load_combined_movies()

# Check if Hindi movies are loaded
hindi_movies = movies_df[movies_df['language'] == 'hindi']
print(f"Hindi movies found: {len(hindi_movies)}")
print("Sample Hindi movies:")
print(hindi_movies[['movieId', 'title']].head(10))

# Test movie matching
def test_movie_match(title):
    clean_title = title.lower().strip()
    matches = movies_df[movies_df['title'].str.lower().str.contains(clean_title, na=False)]
    if not matches.empty:
        print(f"✅ Found '{title}': {matches['title'].values[0]}")
    else:
        print(f"❌ Not found: '{title}'")

# Test some Hindi movie names
test_movie_match("3 Idiots")
test_movie_match("Dangal")
test_movie_match("Lagaan")
test_movie_match("Bahubali")