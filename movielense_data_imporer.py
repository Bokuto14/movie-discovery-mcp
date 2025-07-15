"""
Import MovieLens dataset for real rating data
Download ml-latest-small.zip from https://grouplens.org/datasets/movielens/
"""

import asyncio
import pandas as pd
import os
from datetime import datetime
from src.models.database import movie_db, initialize_database

async def import_movielens(data_path="ml-latest-small"):
    """Import MovieLens dataset"""
    
    await initialize_database()
    
    # Load MovieLens data
    print("📂 Loading MovieLens data...")
    movies_df = pd.read_csv(f"{data_path}/movies.csv")
    ratings_df = pd.read_csv(f"{data_path}/ratings.csv")
    links_df = pd.read_csv(f"{data_path}/links.csv")
    
    print(f"Found {len(movies_df)} movies and {len(ratings_df)} ratings")
    
    # Create MovieLens users
    print("\n👥 Creating MovieLens users...")
    unique_users = ratings_df['userId'].unique()[:100]  # First 100 users
    ml_users = {}
    
    for ml_user_id in unique_users:
        user = await movie_db.create_user(
            username=f"movielens_user_{ml_user_id}",
            email=f"ml_user_{ml_user_id}@movielens.org"
        )
        ml_users[ml_user_id] = user['user_id']
        
    print(f"Created {len(ml_users)} users")
    
    # Import movies with TMDb IDs
    print("\n🎬 Importing movies...")
    movies_added = 0
    
    for _, row in movies_df.iterrows():
        # Get TMDb ID from links
        tmdb_match = links_df[links_df['movieId'] == row['movieId']]
        if tmdb_match.empty or pd.isna(tmdb_match.iloc[0]['tmdbId']):
            continue
            
        tmdb_id = int(tmdb_match.iloc[0]['tmdbId'])
        
        # Parse genres
        genres = [{"name": g.strip()} for g in row['genres'].split('|') if g != '(no genres listed)']
        
        # Extract year from title
        year_match = row['title'][-5:-1]
        release_date = None
        if year_match.isdigit():
            try:
                release_date = datetime(int(year_match), 1, 1).date()
            except:
                pass
        
        movie_data = {
            'id': tmdb_id,
            'title': row['title'][:-7] if row['title'].endswith(')') else row['title'],
            'original_title': row['title'][:-7] if row['title'].endswith(')') else row['title'],
            'release_date': release_date,
            'genres': genres,
            'overview': f"MovieLens movie (ID: {row['movieId']})",
            'vote_average': 0,
            'vote_count': 0,
            'popularity': 0
        }
        
        success = await movie_db.add_or_update_movie(movie_data)
        if success:
            movies_added += 1
            
        if movies_added >= 500:  # Limit for testing
            break
    
    print(f"Added {movies_added} movies")
    
    # Import ratings
    print("\n⭐ Importing ratings...")
    ratings_added = 0
    
    # Get ratings for movies we imported
    for ml_user_id, user_id in ml_users.items():
        user_ratings = ratings_df[ratings_df['userId'] == ml_user_id]
        
        for _, rating in user_ratings.iterrows():
            # Find TMDb ID
            tmdb_match = links_df[links_df['movieId'] == rating['movieId']]
            if tmdb_match.empty or pd.isna(tmdb_match.iloc[0]['tmdbId']):
                continue
                
            tmdb_id = int(tmdb_match.iloc[0]['tmdbId'])
            
            # Add rating (MovieLens uses 0.5-5.0 scale already!)
            success = await movie_db.add_user_rating(
                user_id=user_id,
                movie_id=tmdb_id,
                rating=float(rating['rating']),
                review_text=None
            )
            
            if success:
                ratings_added += 1
                
            if ratings_added % 100 == 0:
                print(f"  Progress: {ratings_added} ratings added...")
    
    await movie_db.close_connection_pool()
    print(f"\n🎉 Import complete!")
    print(f"  Movies: {movies_added}")
    print(f"  Ratings: {ratings_added}")
    print(f"  Users: {len(ml_users)}")

if __name__ == "__main__":
    print("🎬 MovieLens Importer")
    print("=" * 50)
    print("\nMake sure you've downloaded ml-latest-small.zip from:")
    print("https://grouplens.org/datasets/movielens/")
    print("\nExtract it to your project directory.")
    
    data_path = input("\nEnter path to extracted folder (default: ml-latest-small): ").strip()
    if not data_path:
        data_path = "ml-latest-small"
    
    if os.path.exists(data_path):
        asyncio.run(import_movielens(data_path))
    else:
        print(f"❌ Directory '{data_path}' not found!")
        print("Please download and extract the MovieLens dataset first.")