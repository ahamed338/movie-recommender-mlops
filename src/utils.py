# src/utils.py (Simplified - no API needed)
import pandas as pd
import numpy as np

def load_combined_movies():
    """Load BOTH English (original) + Hindi (manual) movies"""
    # Load original English movies (KEEP EXISTING)
    english_movies = pd.read_csv(
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
    english_movies['language'] = 'english'
    
    # Create Hindi movies manually (NEW ADDITION)
    hindi_movies_data = [
        (2001, "Dilwale Dulhania Le Jayenge (1995)"),
        (2002, "3 Idiots (2009)"),
        (2003, "Lagaan (2001)"),
        (2004, "Sholay (1975)"),
        (2005, "Kabhi Khushi Kabhie Gham (2001)"),
        (2006, "Chennai Express (2013)"),
        (2007, "Dangal (2016)"),
        (2008, "Bahubali: The Beginning (2015)"),
        (2009, "PK (2014)"),
        (2010, "Zindagi Na Milegi Dobara (2011)"),
        (2011, "Queen (2013)"),
        (2012, "Barfi! (2012)"),
        (2013, "Gully Boy (2019)"),
        (2014, "Andhadhun (2018)"),
        (2015, "Taare Zameen Par (2007)")
    ]
    
    hindi_movies = pd.DataFrame(hindi_movies_data, columns=["movieId", "title"])
    hindi_movies['language'] = 'hindi'
    
    # COMBINE BOTH DATASETS
    combined_movies = pd.concat([english_movies, hindi_movies], ignore_index=True)
    
    # Create search-friendly titles
    combined_movies["search_title"] = combined_movies["title"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    print(f"✅ Combined dataset: {len(english_movies)} English + {len(hindi_movies)} Hindi = {len(combined_movies)} total movies")
    return combined_movies

def create_hybrid_ratings():
    """Create ratings that include BOTH English and Hindi movies"""
    # Load original English ratings (KEEP EXISTING)
    original_ratings = pd.read_csv(
        "ml-100k/u.data",
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"]
    ).drop(columns=["timestamp"])
    
    # Add synthetic ratings for Hindi movies (NEW ADDITION)
    synthetic_ratings = []
    
    # Hindi movie IDs (2001-2015)
    hindi_movie_ids = list(range(2001, 2016))
    
    # Add ratings for first 100 users
    for user_id in range(1, 101):
        # Each user rates 3-5 random Hindi movies
        num_ratings = np.random.randint(3, 6)
        rated_movies = np.random.choice(hindi_movie_ids, size=num_ratings, replace=False)
        
        for movie_id in rated_movies:
            # Simulate realistic ratings (most ratings are 3-5)
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
            
            synthetic_ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating
            })
    
    # COMBINE BOTH RATINGS
    hybrid_ratings = pd.concat([original_ratings, pd.DataFrame(synthetic_ratings)], ignore_index=True)
    
    print(f"✅ Hybrid ratings: {len(original_ratings)} original + {len(synthetic_ratings)} synthetic = {len(hybrid_ratings)} total ratings")
    return hybrid_ratings

if __name__ == "__main__":
    movies = load_combined_movies()
    ratings = create_hybrid_ratings()
    print("\n📊 Dataset Summary:")
    print(f"Total movies: {len(movies)}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Unique users: {ratings['userId'].nunique()}")
    print(f"Unique movies rated: {ratings['movieId'].nunique()}")