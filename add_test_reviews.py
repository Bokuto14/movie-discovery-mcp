"""
Add test reviews with sentiment for testing
"""

import asyncio
from src.models.database import movie_db, initialize_database

async def add_test_reviews():
    """Add some movie reviews for sentiment analysis testing"""
    
    await initialize_database()
    
    # Sample reviews with different sentiments
    test_reviews = [
        # Positive reviews
        {
            "username": "action_lover",
            "movie_title": "Venom: The Last Dance",
            "rating": 4.5,
            "review": "Absolutely loved this movie! The action sequences were incredible and Tom Hardy's performance was outstanding. The visual effects were mind-blowing. Can't wait for the next one!"
        },
        {
            "username": "rom_com_fan",
            "movie_title": "Sonic the Hedgehog 3",
            "rating": 4.0,
            "review": "Surprisingly heartwarming! Great for families. The humor was on point and the storyline was engaging. Jim Carrey stole the show as always."
        },
        # Mixed reviews
        {
            "username": "sci_fi_geek",
            "movie_title": "Mufasa: The Lion King",
            "rating": 3.0,
            "review": "The visuals were stunning but the story felt a bit predictable. Some parts dragged on while others felt rushed. Still worth watching for the cinematography alone."
        },
        # Negative reviews
        {
            "username": "horror_buff",
            "movie_title": "Venom: The Last Dance",
            "rating": 2.0,
            "review": "Disappointing sequel. The plot was confusing and the pacing was terrible. Too many unnecessary subplots. Expected much more from this franchise."
        },
        {
            "username": "general_viewer",
            "movie_title": "Sonic the Hedgehog 3",
            "rating": 2.5,
            "review": "Not impressed. The acting felt forced and the script was weak. My kids enjoyed it but I found it boring and repetitive."
        },
        # More positive reviews for variety
        {
            "username": "action_lover",
            "movie_title": "Moana 2",
            "rating": 5.0,
            "review": "Disney magic at its finest! Beautiful animation, catchy songs, and a touching story about family and adventure. The music gave me goosebumps!"
        },
        {
            "username": "rom_com_fan",
            "movie_title": "Moana 2",
            "rating": 4.5,
            "review": "A delightful sequel that captures the spirit of the original. The emotional depth surprised me. Great character development and stunning visuals throughout."
        }
    ]
    
    print("🎬 Adding test reviews with sentiment...")
    
    reviews_added = 0
    
    for review_data in test_reviews:
        # Get user
        user = await movie_db.get_user(username=review_data['username'])
        if not user:
            continue
        
        # Find movie by title
        async with movie_db.pool.acquire() as conn:
            movie = await conn.fetchrow(
                "SELECT movie_id FROM movies WHERE title = $1 LIMIT 1",
                review_data['movie_title']
            )
            
            if movie:
                # Add rating with review
                success = await movie_db.add_user_rating(
                    user_id=user['user_id'],
                    movie_id=movie['movie_id'],
                    rating=review_data['rating'],
                    review_text=review_data['review']
                )
                
                if success:
                    reviews_added += 1
                    print(f"✅ Added review from {review_data['username']} for {review_data['movie_title']}")
    
    print(f"\n🎉 Added {reviews_added} test reviews!")
    print("\n💡 You can now test sentiment analysis on these movies:")
    print("- Venom: The Last Dance")
    print("- Sonic the Hedgehog 3")
    print("- Moana 2")
    print("- Mufasa: The Lion King")
    
    await movie_db.close_connection_pool()

if __name__ == "__main__":
    asyncio.run(add_test_reviews())