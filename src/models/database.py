#!/usr/bin/env python3
"""
PostgreSQL Database models for Movie Discovery MCP Server (CORRECTED VERSION)
No tmdb_id references - movie_id IS the TMDB ID
"""

import asyncio
import asyncpg
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class PostgreSQLMovieDatabase:
    """PostgreSQL database handler with corrected schema."""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.pool = None
    
    async def initialize_connection_pool(self):
        """Initialize async connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def close_connection_pool(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def initialize_database(self):
        """Initialize database schema (use the reset script instead)."""
        if not self.pool:
            await self.initialize_connection_pool()
        logger.info("Database connection initialized")
    
    async def create_user(self, username: str, email: str = None) -> Dict:
        """Create a new user and return user data."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            try:
                user = await conn.fetchrow("""
                    INSERT INTO users (username, email, preferences)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (username) DO UPDATE SET
                        updated_at = NOW()
                    RETURNING *
                """, username, email, json.dumps({
                    "favorite_genres": [],
                    "disliked_genres": [],
                    "favorite_actors": [],
                    "favorite_directors": [],
                    "preferred_rating_range": [6.0, 10.0],
                    "preferred_year_range": [1990, 2025],
                    "preferred_runtime_range": [80, 180]
                }))
                
                return dict(user)
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                raise
    
    async def get_user(self, user_id: str = None, username: str = None) -> Optional[Dict]:
        """Get user by ID or username."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            if user_id:
                user = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
            elif username:
                user = await conn.fetchrow("SELECT * FROM users WHERE username = $1", username)
            else:
                return None
            
            return dict(user) if user else None
    
    async def add_or_update_movie(self, movie_data: Dict) -> bool:
        """Add or update movie with comprehensive data."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO movies (
                        movie_id, title, original_title, release_date, overview,
                        genres, vote_average, vote_count, popularity, runtime,
                        budget, revenue, director, cast_members, crew_members,
                        poster_url, backdrop_url, production_companies,
                        production_countries, spoken_languages, keywords
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    ON CONFLICT (movie_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        overview = EXCLUDED.overview,
                        vote_average = EXCLUDED.vote_average,
                        vote_count = EXCLUDED.vote_count,
                        popularity = EXCLUDED.popularity,
                        updated_at = NOW()
                """, 
                    movie_data.get('id'),
                    movie_data.get('title'),
                    movie_data.get('original_title'),
                    movie_data.get('release_date'),
                    movie_data.get('overview'),
                    json.dumps(movie_data.get('genres', [])),
                    movie_data.get('vote_average'),
                    movie_data.get('vote_count'),
                    movie_data.get('popularity'),
                    movie_data.get('runtime'),
                    movie_data.get('budget'),
                    movie_data.get('revenue'),
                    movie_data.get('director'),
                    json.dumps(movie_data.get('cast', [])),
                    json.dumps(movie_data.get('crew', [])),
                    movie_data.get('poster_url'),
                    movie_data.get('backdrop_url'),
                    json.dumps(movie_data.get('production_companies', [])),
                    json.dumps(movie_data.get('production_countries', [])),
                    json.dumps(movie_data.get('spoken_languages', [])),
                    json.dumps(movie_data.get('keywords', []))
                )
                return True
            except Exception as e:
                logger.error(f"Error adding movie: {e}")
                return False
    
    async def add_user_rating(self, user_id: str, movie_id: int, rating: float, 
                             review_text: str = None, is_favorite: bool = False) -> bool:
        """Add or update user rating."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO user_ratings (user_id, movie_id, rating, review_text, is_favorite)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, movie_id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        review_text = EXCLUDED.review_text,
                        is_favorite = EXCLUDED.is_favorite,
                        updated_at = NOW()
                """, user_id, movie_id, rating, review_text, is_favorite)
                
                # Log activity
                await conn.execute("""
                    INSERT INTO user_activity_log (user_id, activity_type, movie_id, metadata)
                    VALUES ($1, 'rating', $2, $3)
                """, user_id, movie_id, json.dumps({
                    'rating': rating,
                    'is_favorite': is_favorite
                }))
                
                return True
            except Exception as e:
                logger.error(f"Error adding rating: {e}")
                return False
    
    async def get_collaborative_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recommendations using collaborative filtering."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            # Find similar users
            similar_users = await conn.fetch("""
                WITH user_similarities AS (
                    SELECT 
                        ur2.user_id,
                        CORR(ur1.rating, ur2.rating) as correlation,
                        COUNT(*) as common_movies
                    FROM user_ratings ur1
                    JOIN user_ratings ur2 ON ur1.movie_id = ur2.movie_id
                    WHERE ur1.user_id = $1 AND ur2.user_id != $1
                    GROUP BY ur2.user_id
                    HAVING COUNT(*) >= 3 AND CORR(ur1.rating, ur2.rating) > 0.3
                    ORDER BY correlation DESC
                    LIMIT 10
                )
                SELECT 
                    m.*,
                    AVG(ur.rating) as predicted_rating,
                    COUNT(ur.rating) as similar_user_count,
                    AVG(us.correlation) as avg_similarity
                FROM user_similarities us
                JOIN user_ratings ur ON us.user_id = ur.user_id
                JOIN movies m ON ur.movie_id = m.movie_id
                WHERE ur.rating >= 4.0
                    AND m.movie_id NOT IN (
                        SELECT movie_id FROM user_ratings WHERE user_id = $1
                    )
                GROUP BY m.movie_id, m.title, m.overview, m.vote_average, 
                         m.popularity, m.release_date, m.genres, m.poster_url
                HAVING COUNT(ur.rating) >= 2
                ORDER BY predicted_rating DESC, avg_similarity DESC
                LIMIT $2
            """, user_id, limit)
            
            return [dict(rec) for rec in similar_users]
    
    async def get_content_based_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recommendations based on user's content preferences."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            # Get user's preferred genres and ratings
            user_preferences = await conn.fetch("""
                SELECT 
                    m.genres,
                    AVG(ur.rating) as avg_rating,
                    COUNT(*) as rating_count
                FROM user_ratings ur
                JOIN movies m ON ur.movie_id = m.movie_id
                WHERE ur.user_id = $1 AND ur.rating >= 4.0
                GROUP BY m.genres
                ORDER BY avg_rating DESC, rating_count DESC
            """, user_id)
            
            if not user_preferences:
                # Fall back to popular movies
                return await self.get_trending_movies(limit)
            
            # Extract favorite genres
            favorite_genres = []
            for pref in user_preferences[:3]:  # Top 3 genre combinations
                if pref['genres']:
                    genres_data = json.loads(pref['genres']) if isinstance(pref['genres'], str) else pref['genres']
                    # Extract just the genre names
                    if isinstance(genres_data, list):
                        for genre in genres_data:
                            if isinstance(genre, dict) and 'name' in genre:
                                favorite_genres.append(genre['name'])
                            elif isinstance(genre, str):
                                favorite_genres.append(genre)
            
            # Find movies with similar genres
            recommendations = await conn.fetch("""
                SELECT DISTINCT m.*,
                    m.vote_average * 0.7 + m.popularity * 0.3 / 1000 as score
                FROM movies m
                WHERE m.genres::jsonb ?| $1
                    AND m.vote_average >= 6.5
                    AND m.vote_count >= 50
                    AND m.movie_id NOT IN (
                        SELECT movie_id FROM user_ratings WHERE user_id = $2
                    )
                ORDER BY score DESC
                LIMIT $3
            """, favorite_genres, user_id, limit)
            
            return [dict(rec) for rec in recommendations]
    
    async def get_trending_movies(self, limit: int = 20, time_period: str = 'week') -> List[Dict]:
        """Get trending movies based on recent activity."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            if time_period == 'day':
                interval = "1 day"
            else:
                interval = "7 days"
            
            movies = await conn.fetch(f"""
                SELECT m.*,
                    COALESCE(recent_activity.activity_score, 0) + 
                    (m.popularity / 1000) + 
                    (m.vote_average / 2) as trending_score
                FROM movies m
                LEFT JOIN (
                    SELECT 
                        movie_id,
                        COUNT(*) * 10 as activity_score
                    FROM user_activity_log
                    WHERE created_at >= NOW() - INTERVAL '{interval}'
                        AND activity_type IN ('rating', 'view')
                    GROUP BY movie_id
                ) recent_activity ON m.movie_id = recent_activity.movie_id
                WHERE m.vote_average >= 6.0 AND m.vote_count >= 100
                ORDER BY trending_score DESC
                LIMIT $1
            """, limit)
            
            return [dict(movie) for movie in movies]
    
    async def get_user_statistics(self, user_id: str) -> Dict:
        """Get comprehensive user statistics."""
        if not self.pool:
            await self.initialize_connection_pool()
            
        async with self.pool.acquire() as conn:
            # Basic rating stats
            basic_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating,
                    COUNT(CASE WHEN is_favorite THEN 1 END) as favorites_count
                FROM user_ratings
                WHERE user_id = $1
            """, user_id)
            
            # Genre preferences
            genre_stats = await conn.fetch("""
                SELECT 
                    genre,
                    AVG(ur.rating) as avg_rating,
                    COUNT(*) as count
                FROM user_ratings ur
                JOIN movies m ON ur.movie_id = m.movie_id,
                LATERAL jsonb_array_elements_text(m.genres) as genre
                WHERE ur.user_id = $1
                GROUP BY genre
                HAVING COUNT(*) >= 2
                ORDER BY avg_rating DESC, count DESC
                LIMIT 10
            """, user_id)
            
            # Recent activity
            recent_activity = await conn.fetch("""
                SELECT activity_type, COUNT(*) as count
                FROM user_activity_log
                WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY activity_type
            """, user_id)
            
            return {
                "basic_stats": dict(basic_stats) if basic_stats else {},
                "genre_preferences": [dict(g) for g in genre_stats],
                "recent_activity": [dict(a) for a in recent_activity]
            }

# Global database instance
movie_db = PostgreSQLMovieDatabase()

async def initialize_database():
    """Initialize the database connection."""
    await movie_db.initialize_connection_pool()
    logger.info("PostgreSQL Movie database initialized")

async def cleanup_database():
    """Clean up database connections."""
    await movie_db.close_connection_pool()

if __name__ == "__main__":
    # Test the database
    async def test_db():
        await initialize_database()
        print("Database connection test completed!")
        await cleanup_database()
    
    asyncio.run(test_db())