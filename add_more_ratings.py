"""
Quick script to add more overlapping ratings for better ML recommendations
"""

import asyncio
import random
from src.models.database import movie_db, initialize_database

async def add_overlapping_ratings():
    """Add more ratings with guaranteed overlap between users"""
    
    await initialize_database()
    
    # Get all users
    users = []
    for username in ["action_lover", "rom_com_fan", "sci_fi_geek", "horror_buff", "general_viewer"]:
        user = await movie_db.get_user(username=username)
        if user:
            users.append(user)
    
    # Get top 50 most popular movies (these will have overlap)
    popular_movies = await movie_db.get_trending_movies(limit=50)
    
    print(f"Adding overlapping ratings for {len(users)} users on {len(popular_movies)} popular movies...")
    
    ratings_added = 0
    
    # Ensure each user rates at least 70% of popular movies
    for user in users:
        movies_to_rate = random.sample(popular_movies, int(len(popular_movies) * 0.7))
        
        for movie in movies_to_rate:
            # Check if already rated
            existing = await movie_db.pool.acquire()
            try:
                result = await existing.fetchrow(
                    "SELECT * FROM user_ratings WHERE user_id = $1 AND movie_id = $2",
                    user['user_id'], movie['movie_id']
                )
                
                if not result:  # Not rated yet
                    # Generate rating based on user type
                    base_rating = float(movie.get('vote_average', 7.0)) / 2
                    
                    # Add user preference bias
                    if user['username'] == 'action_lover' and 'Action' in str(movie.get('genres', '')):
                        base_rating += random.uniform(0.5, 1.0)
                    elif user['username'] == 'rom_com_fan' and ('Romance' in str(movie.get('genres', '')) or 'Comedy' in str(movie.get('genres', ''))):
                        base_rating += random.uniform(0.5, 1.0)
                    # ... similar for other users
                    
                    final_rating = max(0.5, min(5.0, base_rating + random.uniform(-0.5, 0.5)))
                    final_rating = round(final_rating * 2) / 2
                    
                    success = await movie_db.add_user_rating(
                        user_id=user['user_id'],
                        movie_id=movie['movie_id'],
                        rating=final_rating
                    )
                    
                    if success:
                        ratings_added += 1
            finally:
                await existing.close()
    
    print(f"✅ Added {ratings_added} additional ratings!")
    
    # Test recommendations again
    print("\n🧪 Testing recommendations after adding more data...")
    
    for user in users[:2]:  # Test first 2 users
        recs = await movie_db.get_collaborative_recommendations(user['user_id'], limit=5)
        print(f"\n{user['username']} now has {len(recs)} collaborative recommendations")
        
    await movie_db.close_connection_pool()

if __name__ == "__main__":
    asyncio.run(add_overlapping_ratings())