"""
Bulk Movie Importer - Add hundreds of movies to your database
"""

import asyncio
import requests
from datetime import datetime
from src.models.database import movie_db, initialize_database
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

async def import_movies_by_category():
    """Import movies from various categories"""
    
    await initialize_database()
    
    categories = [
        {"name": "Popular Movies", "endpoint": "movie/popular", "pages": 5},
        {"name": "Top Rated Movies", "endpoint": "movie/top_rated", "pages": 5},
        {"name": "Now Playing", "endpoint": "movie/now_playing", "pages": 3},
        {"name": "Upcoming Movies", "endpoint": "movie/upcoming", "pages": 2},
    ]
    
    # Genre-specific discoveries
    genres = [
        {"id": 28, "name": "Action"},
        {"id": 35, "name": "Comedy"},
        {"id": 18, "name": "Drama"},
        {"id": 27, "name": "Horror"},
        {"id": 878, "name": "Science Fiction"},
        {"id": 10749, "name": "Romance"},
        {"id": 53, "name": "Thriller"},
        {"id": 16, "name": "Animation"}
    ]
    
    total_added = 0
    
    # Import from categories
    for category in categories:
        print(f"\n📂 Importing {category['name']}...")
        
        for page in range(1, category['pages'] + 1):
            url = f"{TMDB_BASE_URL}/{category['endpoint']}"
            params = {
                'api_key': TMDB_API_KEY,
                'page': page
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            for movie in data.get('results', []):
                # Get detailed info
                details_url = f"{TMDB_BASE_URL}/movie/{movie['id']}"
                details_response = requests.get(details_url, params={'api_key': TMDB_API_KEY})
                movie_details = details_response.json()
                
                # Prepare movie data
                movie_data = {
                    'id': movie['id'],
                    'title': movie['title'],
                    'original_title': movie.get('original_title', movie['title']),
                    'release_date': None,
                    'overview': movie.get('overview', ''),
                    'genres': [{'id': g['id'], 'name': g['name']} for g in movie_details.get('genres', [])],
                    'vote_average': movie.get('vote_average', 0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0),
                    'runtime': movie_details.get('runtime'),
                    'budget': movie_details.get('budget', 0),
                    'revenue': movie_details.get('revenue', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
                    'backdrop_url': f"https://image.tmdb.org/t/p/w500{movie.get('backdrop_path')}" if movie.get('backdrop_path') else None,
                }
                
                # Parse date properly
                if movie.get('release_date'):
                    try:
                        movie_data['release_date'] = datetime.strptime(movie['release_date'], '%Y-%m-%d').date()
                    except:
                        pass
                
                # Add to database
                success = await movie_db.add_or_update_movie(movie_data)
                if success:
                    total_added += 1
                    print(f"✅ Added: {movie['title']}")
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
    
    # Import by genre
    print("\n🎭 Importing movies by genre...")
    
    for genre in genres:
        print(f"\n  Genre: {genre['name']}")
        
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre['id'],
            'sort_by': 'popularity.desc',
            'page': 1
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        for movie in data.get('results', [])[:10]:  # Top 10 per genre
            # Similar processing as above...
            movie_data = {
                'id': movie['id'],
                'title': movie['title'],
                'original_title': movie.get('original_title', movie['title']),
                'release_date': None,
                'overview': movie.get('overview', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'popularity': movie.get('popularity', 0),
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
            }
            
            if movie.get('release_date'):
                try:
                    movie_data['release_date'] = datetime.strptime(movie['release_date'], '%Y-%m-%d').date()
                except:
                    pass
            
            success = await movie_db.add_or_update_movie(movie_data)
            if success:
                total_added += 1
            
            await asyncio.sleep(0.1)
    
    await movie_db.close_connection_pool()
    print(f"\n🎉 Import complete! Added {total_added} movies to database.")

async def search_and_import(search_queries):
    """Import movies based on search queries"""
    
    await initialize_database()
    total_added = 0
    
    for query in search_queries:
        print(f"\n🔍 Searching for: {query}")
        
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': 1
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        for movie in data.get('results', [])[:20]:  # Top 20 results
            movie_data = {
                'id': movie['id'],
                'title': movie['title'],
                'original_title': movie.get('original_title', movie['title']),
                'release_date': None,
                'overview': movie.get('overview', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'popularity': movie.get('popularity', 0),
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
            }
            
            if movie.get('release_date'):
                try:
                    movie_data['release_date'] = datetime.strptime(movie['release_date'], '%Y-%m-%d').date()
                except:
                    pass
            
            success = await movie_db.add_or_update_movie(movie_data)
            if success:
                total_added += 1
                print(f"  ✅ {movie['title']}")
            
            await asyncio.sleep(0.1)
    
    await movie_db.close_connection_pool()
    print(f"\n🎉 Added {total_added} movies from searches!")

if __name__ == "__main__":
    print("🎬 Bulk Movie Importer")
    print("=" * 50)
    print("\nChoose import method:")
    print("1. Import by categories (Popular, Top Rated, etc.)")
    print("2. Import by search terms")
    print("3. Import everything (categories + specific searches)")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        asyncio.run(import_movies_by_category())
    elif choice == "2":
        # Add your search terms here
        searches = [
            "Marvel", "DC Comics", "Star Wars", "Harry Potter",
            "Lord of the Rings", "James Bond", "Mission Impossible",
            "Fast and Furious", "Pixar", "Studio Ghibli",
            "Christopher Nolan", "Quentin Tarantino", "Steven Spielberg",
            "Oscar winner", "Cannes", "Academy Award"
        ]
        asyncio.run(search_and_import(searches))
    elif choice == "3":
        asyncio.run(import_movies_by_category())
        searches = ["Marvel", "Star Wars", "Harry Potter", "Pixar"]
        asyncio.run(search_and_import(searches))
    else:
        print("Invalid choice!")