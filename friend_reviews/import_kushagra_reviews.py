"""
Import script for Kushagra's movie reviews
Generated on: 2025-07-16T00:17:48.780390
Total reviews: 25
"""

import asyncio
import sys
import os

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.database import movie_db, initialize_database
except ImportError:
    sys.path.append('.')
    from models.database import movie_db, initialize_database

async def import_reviews():
    """Import Kushagra's reviews into the database"""
    print(f"🎬 Importing reviews from Kushagra...")
    
    try:
        # Initialize database
        await initialize_database()
        
        # Create or get user
        user = await movie_db.create_user(
            username="kushagra",
            email="kushagra@friend.local"
        )
        
        print(f"👤 User created/found: {user['username']}")
        
        # Review data
        reviews = [
        {
                "movie_id": 129,
                "movie_title": "Spirited Away",
                "rating": 5,
                "review": "Greatest animated movie of all time"
        },
        {
                "movie_id": 155,
                "movie_title": "The Dark Knight",
                "rating": 5,
                "review": "Greatest superhero movie of all time"
        },
        {
                "movie_id": 164,
                "movie_title": "Breakfast at Tiffany's",
                "rating": 3,
                "review": "Decent"
        },
        {
                "movie_id": 238,
                "movie_title": "The Godfather",
                "rating": 5,
                "review": "2nd greatest movie of all time"
        },
        {
                "movie_id": 240,
                "movie_title": "The Godfather Part II",
                "rating": 5,
                "review": "Greatest movie of all time"
        },
        {
                "movie_id": 274,
                "movie_title": "The Silence of the Lambs",
                "rating": 5,
                "review": "Wow!! Absolutely amazing"
        },
        {
                "movie_id": 311,
                "movie_title": "Once Upon a Time in America",
                "rating": 4,
                "review": None
        },
        {
                "movie_id": 389,
                "movie_title": "12 Angry Men",
                "rating": 5,
                "review": "Never thought that a movie about12 people talking would be this good"
        },
        {
                "movie_id": 423,
                "movie_title": "The Pianist",
                "rating": 5,
                "review": "adrian brody at his best. What a performance. what an amazIng movie"
        },
        {
                "movie_id": 429,
                "movie_title": "The Good, the Bad and the Ugly",
                "rating": 5,
                "review": "Perfection"
        },
        {
                "movie_id": 497,
                "movie_title": "The Green Mile",
                "rating": 4,
                "review": "One of the best tom hanks performance"
        },
        {
                "movie_id": 567,
                "movie_title": "Rear Window",
                "rating": 3,
                "review": None
        },
        {
                "movie_id": 581,
                "movie_title": "Dances with Wolves",
                "rating": 3,
                "review": None
        },
        {
                "movie_id": 599,
                "movie_title": "Sunset Boulevard",
                "rating": 2,
                "review": None
        },
        {
                "movie_id": 935,
                "movie_title": "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb",
                "rating": 1,
                "review": "Not seen"
        },
        {
                "movie_id": 1891,
                "movie_title": "The Empire Strikes Back",
                "rating": 5,
                "review": "Star wars at its peak."
        },
        {
                "movie_id": 12477,
                "movie_title": "Grave of the Fireflies",
                "rating": 4,
                "review": "So sad but so good"
        },
        {
                "movie_id": 77338,
                "movie_title": "The Intouchables",
                "rating": 4,
                "review": "Pretty pretty good"
        },
        {
                "movie_id": 299536,
                "movie_title": "Avengers: Infinity War",
                "rating": 4,
                "review": "my 2nd favorite marvel movie of all time"
        },
        {
                "movie_id": 330459,
                "movie_title": "Rogue One: A Star Wars Story",
                "rating": 4,
                "review": "My favorite recent star wars movies"
        },
        {
                "movie_id": 803796,
                "movie_title": "KPop Demon Hunters",
                "rating": 4,
                "review": "Surprisingly very funny"
        },
        {
                "movie_id": 950396,
                "movie_title": "The Gorge",
                "rating": 3,
                "review": "Decent evening watch"
        },
        {
                "movie_id": 1376434,
                "movie_title": "Predator: Killer of Killers",
                "rating": 4,
                "review": None
        },
        {
                "movie_id": 1412113,
                "movie_title": "Squid Game: Making Season 2",
                "rating": 1,
                "review": "Just a documentary"
        },
        {
                "movie_id": 1426776,
                "movie_title": "STRAW",
                "rating": 3,
                "review": "Good one time watch"
        }
]
        
        success_count = 0
        error_count = 0
        
        for review in reviews:
            try:
                success = await movie_db.add_user_rating(
                    user_id=user['user_id'],
                    movie_id=review['movie_id'],
                    rating=float(review['rating']),
                    review_text=review.get('review') if review.get('review') else None
                )
                
                if success:
                    success_count += 1
                    print(f"✅ {review['movie_title']} - {review['rating']}/5")
                else:
                    error_count += 1
                    print(f"❌ Failed: {review['movie_title']}")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ Error with {review.get('movie_title', 'Unknown')}: {e}")
        
        print(f"\n🎉 Import Complete!")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   📊 Total: {len(reviews)}")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await movie_db.close_connection_pool()

if __name__ == "__main__":
    asyncio.run(import_reviews())
