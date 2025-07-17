# fix_all_genres.py
import asyncio
import sys
import os
import httpx
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.database import movie_db, initialize_database

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

async def fix_all_genres():
    print("🔧 Fixing ALL movie genres using TMDB API...")
    await initialize_database()
    
    client = httpx.AsyncClient()
    fixed_count = 0
    error_count = 0
    
    try:
        async with movie_db.pool.acquire() as conn:
            # Get movies with NULL genres
            movies = await conn.fetch("SELECT movie_id, title FROM movies WHERE genres IS NULL")
            print(f"Found {len(movies)} movies to fix...")
            
            for i, movie in enumerate(movies, 1):
                try:
                    # Fetch from TMDB
                    url = f"{TMDB_BASE_URL}/movie/{movie['movie_id']}"
                    params = {"api_key": TMDB_API_KEY}
                    
                    response = await client.get(url, params=params)
                    if response.status_code == 404:
                        print(f"{i:3d}. ❌ {movie['title']} - Not found on TMDB")
                        error_count += 1
                        continue
                    
                    response.raise_for_status()
                    movie_data = response.json()
                    genres = movie_data.get('genres', [])
                    
                    # Update database
                    await conn.execute("""
                        UPDATE movies SET genres = $1 WHERE movie_id = $2
                    """, json.dumps(genres), movie['movie_id'])
                    
                    fixed_count += 1
                    genre_names = [g['name'] for g in genres]
                    print(f"{i:3d}. ✅ {movie['title']} - {genre_names}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    error_count += 1
                    print(f"{i:3d}. ❌ {movie['title']} - Error: {e}")
    
    finally:
        await client.aclose()
        await movie_db.close_connection_pool()
    
    print(f"\n🎉 Complete! Fixed: {fixed_count}, Errors: {error_count}")

if __name__ == "__main__":
    asyncio.run(fix_all_genres())