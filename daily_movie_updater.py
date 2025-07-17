# daily_movie_updater.py - Adds new popular movies daily
import asyncio
import sys
import os
import httpx
import json
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.database import movie_db, initialize_database

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

async def add_daily_movies():
    print(f"🎬 Daily Movie Update - {datetime.now().strftime('%Y-%m-%d')}")
    await initialize_database()
    
    client = httpx.AsyncClient()
    added_count = 0
    
    try:
        # Get trending, popular, and top-rated movies
        endpoints = [
            "trending/movie/day",
            "trending/movie/week", 
            "movie/popular",
            "movie/top_rated",
            "movie/now_playing",
            "movie/upcoming"
        ]
        
        all_movie_ids = set()
        
        for endpoint in endpoints:
            try:
                # Get multiple pages
                for page in [1, 2, 3]:
                    url = f"{TMDB_BASE_URL}/{endpoint}"
                    params = {"api_key": TMDB_API_KEY, "page": page}
                    
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    for movie in data.get('results', []):
                        all_movie_ids.add(movie['id'])
                    
                    await asyncio.sleep(0.1)  # Rate limiting
                    
            except Exception as e:
                print(f"❌ Error fetching {endpoint}: {e}")
        
        print(f"Found {len(all_movie_ids)} unique movies to check...")
        
        async with movie_db.pool.acquire() as conn:
            for i, movie_id in enumerate(all_movie_ids, 1):
                try:
                    # Check if movie already exists
                    exists = await conn.fetchval(
                        "SELECT 1 FROM movies WHERE movie_id = $1", movie_id
                    )
                    
                    if exists:
                        continue  # Skip if already in database
                    
                    # Fetch detailed movie data
                    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
                    params = {
                        "api_key": TMDB_API_KEY,
                        "append_to_response": "credits,keywords"
                    }
                    
                    response = await client.get(url, params=params)
                    if response.status_code == 404:
                        continue
                    
                    response.raise_for_status()
                    movie_data = response.json()
                    
                    # Parse release date
                    release_date = None
                    if movie_data.get('release_date'):
                        try:
                            release_date = datetime.strptime(
                                movie_data['release_date'], '%Y-%m-%d'
                            ).date()
                        except:
                            pass
                    
                    # Prepare movie data
                    enhanced_movie = {
                        'id': movie_data['id'],
                        'title': movie_data['title'],
                        'original_title': movie_data.get('original_title'),
                        'overview': movie_data.get('overview'),
                        'release_date': release_date,
                        'vote_average': movie_data.get('vote_average', 0),
                        'vote_count': movie_data.get('vote_count', 0),
                        'popularity': movie_data.get('popularity', 0),
                        'genres': movie_data.get('genres', []),
                        'runtime': movie_data.get('runtime'),
                        'budget': movie_data.get('budget', 0),
                        'revenue': movie_data.get('revenue', 0)
                    }
                    
                    # Insert into database
                    await conn.execute("""
                        INSERT INTO movies (
                            movie_id, title, original_title, overview, release_date,
                            vote_average, vote_count, popularity, genres, runtime, budget, revenue
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (movie_id) DO NOTHING
                    """, 
                        movie_data['id'],
                        movie_data['title'],
                        movie_data.get('original_title'),
                        movie_data.get('overview'),
                        release_date,
                        movie_data.get('vote_average', 0),
                        movie_data.get('vote_count', 0),
                        movie_data.get('popularity', 0),
                        json.dumps(movie_data.get('genres', [])),
                        movie_data.get('runtime'),
                        movie_data.get('budget', 0),
                        movie_data.get('revenue', 0)
                    )
                    
                    added_count += 1
                    print(f"{added_count:3d}. ✅ Added: {movie_data['title']}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                    if i % 50 == 0:
                        print(f"   📊 Progress: {i}/{len(all_movie_ids)} checked, {added_count} added")
                
                except Exception as e:
                    print(f"❌ Error processing movie {movie_id}: {e}")
    
    finally:
        await client.aclose()
        await movie_db.close_connection_pool()
    
    print(f"\n🎉 Daily update complete! Added {added_count} new movies")
    
    # Show total database stats
    await initialize_database()
    async with movie_db.pool.acquire() as conn:
        total_movies = await conn.fetchval("SELECT COUNT(*) FROM movies")
        total_ratings = await conn.fetchval("SELECT COUNT(*) FROM user_ratings")
    await movie_db.close_connection_pool()
    
    print(f"📊 Database now has: {total_movies} movies, {total_ratings} ratings")

if __name__ == "__main__":
    asyncio.run(add_daily_movies())