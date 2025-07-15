"""
Movie Database Population and Testing Script (Corrected)
=======================================================
This script works with your asyncpg-based database structure
"""

import asyncio
import random
from datetime import datetime, date
from decimal import Decimal
import json
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Import your database module
from src.models.database import movie_db, initialize_database

# TMDB setup
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_tmdb_movies(page=1):
    """Fetch popular movies from TMDB"""
    url = f"{TMDB_BASE_URL}/movie/popular"
    params = {
        'api_key': TMDB_API_KEY,
        'page': page
    }
    response = requests.get(url, params=params)
    return response.json()

def get_movie_details(movie_id):
    """Get detailed movie info including cast and crew"""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,keywords'
    }
    response = requests.get(url, params=params)
    return response.json()

async def populate_movies(num_movies=100):
    """Populate database with movies from TMDB"""
    print(f"🎬 Fetching {num_movies} popular movies from TMDB...")
    
    movies_added = 0
    page = 1
    
    while movies_added < num_movies:
        data = get_tmdb_movies(page)
        
        for movie_data in data['results']:
            try:
                # Get detailed movie info
                detailed_movie = get_movie_details(movie_data['id'])
                
                # Parse release date
                release_date = None
                if movie_data.get('release_date'):
                    try:
                        release_date = datetime.strptime(movie_data['release_date'], '%Y-%m-%d').date()
                    except ValueError:
                        release_date = None
                
                # Prepare movie data for database
                movie_to_add = {
                    'id': movie_data['id'],
                    'title': movie_data['title'],
                    'original_title': movie_data.get('original_title', movie_data['title']),
                    'release_date': release_date,
                    'overview': movie_data.get('overview', ''),
                    'genres': [{'id': g['id'], 'name': g['name']} for g in detailed_movie.get('genres', [])],
                    'vote_average': movie_data.get('vote_average', 0),
                    'vote_count': movie_data.get('vote_count', 0),
                    'popularity': movie_data.get('popularity', 0),
                    'runtime': detailed_movie.get('runtime'),
                    'budget': detailed_movie.get('budget', 0),
                    'revenue': detailed_movie.get('revenue', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
                    'backdrop_url': f"https://image.tmdb.org/t/p/w500{movie_data.get('backdrop_path')}" if movie_data.get('backdrop_path') else None,
                    'production_companies': [{'id': c['id'], 'name': c['name']} for c in detailed_movie.get('production_companies', [])],
                    'production_countries': detailed_movie.get('production_countries', []),
                    'spoken_languages': detailed_movie.get('spoken_languages', []),
                    'keywords': [{'id': k['id'], 'name': k['name']} for k in detailed_movie.get('keywords', {}).get('keywords', [])]
                }
                
                # Extract director and cast
                credits = detailed_movie.get('credits', {})
                crew = credits.get('crew', [])
                cast = credits.get('cast', [])
                
                # Find director
                directors = [c for c in crew if c.get('job') == 'Director']
                movie_to_add['director'] = directors[0]['name'] if directors else None
                
                # Get top cast members
                movie_to_add['cast'] = [
                    {
                        'id': actor['id'],
                        'name': actor['name'],
                        'character': actor.get('character', ''),
                        'order': actor.get('order', 999)
                    }
                    for actor in cast[:10]  # Top 10 cast members
                ]
                
                # Get key crew members
                movie_to_add['crew'] = [
                    {
                        'id': person['id'],
                        'name': person['name'],
                        'job': person['job'],
                        'department': person.get('department', '')
                    }
                    for person in crew[:10]  # Top 10 crew members
                ]
                
                # Add movie to database
                success = await movie_db.add_or_update_movie(movie_to_add)
                if success:
                    movies_added += 1
                    print(f"✅ Added: {movie_to_add['title']}")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error processing movie {movie_data.get('title', 'Unknown')}: {str(e)}")
                continue
            
            if movies_added >= num_movies:
                break
        
        page += 1
        if page > 5:  # Safety limit
            break
    
    print(f"\n🎉 Successfully added {movies_added} movies to database!")
    return movies_added

async def create_mock_users():
    """Create different types of mock users with distinct preferences"""
    print("\n👥 Creating mock users...")
    
    # Define user personas
    user_profiles = [
        {
            "username": "action_lover",
            "email": "action@test.com",
            "preferences": {
                "favorite_genres": ["Action", "Thriller", "Crime"],
                "preferred_rating_range": [6.5, 10.0],
                "preferred_runtime_range": [90, 150]
            },
            "rating_behavior": "high_for_action"
        },
        {
            "username": "rom_com_fan",
            "email": "romcom@test.com", 
            "preferences": {
                "favorite_genres": ["Romance", "Comedy"],
                "preferred_rating_range": [7.0, 10.0],
                "preferred_runtime_range": [85, 120]
            },
            "rating_behavior": "high_for_romance"
        },
        {
            "username": "sci_fi_geek",
            "email": "scifi@test.com",
            "preferences": {
                "favorite_genres": ["Science Fiction", "Fantasy", "Adventure"],
                "preferred_rating_range": [7.5, 10.0],
                "preferred_runtime_range": [100, 180]
            },
            "rating_behavior": "high_for_scifi"
        },
        {
            "username": "horror_buff",
            "email": "horror@test.com",
            "preferences": {
                "favorite_genres": ["Horror", "Thriller", "Mystery"],
                "preferred_rating_range": [6.0, 10.0],
                "preferred_runtime_range": [80, 120]
            },
            "rating_behavior": "high_for_horror"
        },
        {
            "username": "general_viewer",
            "email": "general@test.com",
            "preferences": {
                "favorite_genres": ["Drama", "Comedy", "Action"],
                "preferred_rating_range": [6.5, 10.0],
                "preferred_runtime_range": [90, 140]
            },
            "rating_behavior": "balanced"
        }
    ]
    
    created_users = []
    
    for profile in user_profiles:
        # Create user with preferences
        user = await movie_db.create_user(
            username=profile['username'],
            email=profile['email']
        )
        
        # Update user preferences
        if user:
            # Get existing preferences and update them
            current_prefs = json.loads(user.get('preferences', '{}'))
            current_prefs.update(profile['preferences'])
            
            # Update in database (you might need to add this method to your database.py)
            # For now, we'll just store the user with behavior info
            created_users.append((user, profile['rating_behavior']))
            print(f"✅ Created user: {profile['username']}")
    
    return created_users

async def generate_ratings(users_with_behavior):
    """Generate realistic ratings based on user preferences"""
    print("\n⭐ Generating ratings...")
    
    # Get all movies from database
    # Since we don't have a direct get_all_movies method, we'll use trending as a proxy
    movies = await movie_db.get_trending_movies(limit=100)
    
    # Genre preferences for rating behavior
    genre_preferences = {
        "high_for_action": ["Action", "Thriller", "Crime"],
        "high_for_romance": ["Romance", "Comedy", "Drama"],
        "high_for_scifi": ["Science Fiction", "Fantasy", "Adventure"],
        "high_for_horror": ["Horror", "Thriller", "Mystery"],
        "balanced": []  # No specific preferences
    }
    
    ratings_added = 0
    
    for user_data in users_with_behavior:
        if isinstance(user_data, tuple):
            user, behavior = user_data
        else:
            user = user_data
            behavior = "balanced"
        
        # Each user rates 30-60 movies
        num_ratings = random.randint(30, 60)
        movies_to_rate = random.sample(movies, min(num_ratings, len(movies)))
        
        for movie in movies_to_rate:
            # Parse movie genres
            movie_genres = []
            if movie.get('genres'):
                genres_data = json.loads(movie['genres']) if isinstance(movie['genres'], str) else movie['genres']
                movie_genres = [g['name'] for g in genres_data if isinstance(g, dict)]
            
            # Base rating on movie's TMDB rating
            vote_avg = movie.get('vote_average', 7.0)
            base_rating = float(vote_avg) / 2 if vote_avg else 3.5  # Convert from 10-scale to 5-scale
            
            # Add personal preference bias
            preference_boost = 0
            if behavior in genre_preferences and genre_preferences[behavior]:
                for genre in movie_genres:
                    if genre in genre_preferences[behavior]:
                        preference_boost = random.uniform(0.5, 1.5)
                        break
            
            # Calculate final rating with some randomness
            final_rating = base_rating + preference_boost + random.uniform(-0.5, 0.5)
            final_rating = max(0.5, min(5.0, final_rating))  # Clamp to 0.5-5.0
            final_rating = round(final_rating * 2) / 2  # Round to nearest 0.5
            
            # Add review text for some highly rated movies
            review_text = None
            if final_rating >= 4.0 and random.random() < 0.3:
                reviews = [
                    "Amazing movie! Highly recommend.",
                    "One of my favorites!",
                    "Fantastic film, worth watching.",
                    "Really enjoyed this one.",
                    "Great story and acting!"
                ]
                review_text = random.choice(reviews)
            
            # Add rating
            success = await movie_db.add_user_rating(
                user_id=user['user_id'],
                movie_id=movie['movie_id'],
                rating=final_rating,
                review_text=review_text,
                is_favorite=(final_rating >= 4.5 and random.random() < 0.5)
            )
            
            if success:
                ratings_added += 1
    
    print(f"✅ Generated {ratings_added} ratings!")
    return ratings_added

async def test_recommendations():
    """Test the personalized recommendation system"""
    print("\n🎯 Testing personalized recommendations...")
    
    # Get users
    test_users = ["action_lover", "rom_com_fan", "sci_fi_geek", "horror_buff", "general_viewer"]
    
    for username in test_users:
        # Get user
        user = await movie_db.get_user(username=username)
        if not user:
            continue
            
        print(f"\n📊 Analysis for {username}:")
        
        # Get user statistics
        stats = await movie_db.get_user_statistics(user['user_id'])
        
        if stats['basic_stats']:
            basic = stats['basic_stats']
            avg_rating_raw = basic.get('avg_rating', 0)
            avg_rating = float(avg_rating_raw) if avg_rating_raw and isinstance(avg_rating_raw, Decimal) else avg_rating_raw or 0
            print(f"   - Total ratings: {basic.get('total_ratings', 0)}")
            print(f"   - Average rating: {avg_rating:.1f}/5.0")
            print(f"   - Favorites: {basic.get('favorites_count', 0)}")
        
        if stats['genre_preferences']:
            print(f"   - Top genres:")
            for genre in stats['genre_preferences'][:3]:
                avg_rating_raw = genre.get('avg_rating', 0)
                avg_rating = float(avg_rating_raw) if avg_rating_raw and isinstance(avg_rating_raw, Decimal) else avg_rating_raw or 0
                print(f"     • {genre['genre']}: {avg_rating:.1f}/5.0 ({genre['count']} movies)")
        
        # Get recommendations
        recommendations = await movie_db.get_collaborative_recommendations(user['user_id'], limit=5)
        if recommendations:
            print(f"   - Top recommendations:")
            for i, movie in enumerate(recommendations[:5], 1):
                predicted_rating_raw = movie.get('predicted_rating', 0)
                predicted_rating = float(predicted_rating_raw) if predicted_rating_raw and isinstance(predicted_rating_raw, Decimal) else predicted_rating_raw or 0
                print(f"     {i}. {movie['title']} (predicted: {predicted_rating:.1f}/5.0)")

async def main():
    """Main execution function"""
    print("🚀 Movie Discovery Database Setup")
    print("=" * 50)
    
    try:
        # Initialize database connection
        await initialize_database()
        
        # Step 1: Populate movies
        await populate_movies(100)
        
        # Step 2: Create mock users
        users = await create_mock_users()
        
        # Step 3: Generate ratings
        await generate_ratings(users)
        
        # Step 4: Test recommendations
        await test_recommendations()
        
        print("\n🎉 Setup complete! Your database is ready for testing.")
        print("\n💡 Next steps:")
        print("1. Start your MCP server: python -m src.server")
        print("2. In Claude Desktop, try these commands:")
        print("   - 'Get personalized recommendations for action_lover'")
        print("   - 'Rate a movie for sci_fi_geek'")
        print("   - 'Analyze preferences for rom_com_fan'")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await movie_db.close_connection_pool()

if __name__ == "__main__":
    asyncio.run(main())