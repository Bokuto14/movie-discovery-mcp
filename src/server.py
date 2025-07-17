#!/usr/bin/env python3
"""
Enhanced AI-Powered Movie Discovery MCP Server with ML Features
Includes PostgreSQL database, collaborative filtering, and advanced recommendations.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import re
from typing import List, Dict, Optional, Tuple
from textblob import TextBlob
#import spacy
from datetime import datetime

import httpx
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Import our database models
import sys
sys.path.append('.')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.database import movie_db, initialize_database, cleanup_database
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
app = Server("movie-discovery-mcp-ml")

# TMDB API configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

class EnhancedMovieDiscoveryServer:
    """Enhanced server class with ML-powered recommendations."""
    
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.default_user = "anonymous_user"  # Default user for demo
        self.mood_analyzer = MoodAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()

    async def initialize(self):
        """Initialize the server and database."""
        await initialize_database()
        logger.info("Enhanced Movie Discovery Server initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()
        await cleanup_database()
    
    async def fetch_tmdb_data(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from TMDB API with enhanced error handling."""
        if not TMDB_API_KEY:
            raise ValueError("TMDB API key not configured. Please set TMDB_API_KEY environment variable.")
        
        if params is None:
            params = {}
        params["api_key"] = TMDB_API_KEY
        
        url = f"{TMDB_BASE_URL}/{endpoint}"
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"TMDB API request failed: {e}")
            raise
    
    async def get_enhanced_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """Get comprehensive movie details and store in database."""
        # Fetch from TMDB
        movie_data = await self.fetch_tmdb_data(f"movie/{movie_id}")
        credits_data = await self.fetch_tmdb_data(f"movie/{movie_id}/credits")
        keywords_data = await self.fetch_tmdb_data(f"movie/{movie_id}/keywords")
        
        # Enhance with additional data
        enhanced_movie = {
            "id": movie_data.get("id"),
            "title": movie_data.get("title"),
            "original_title": movie_data.get("original_title"),
            "overview": movie_data.get("overview"),
            "release_date": movie_data.get("release_date"),
            "runtime": movie_data.get("runtime"),
            "vote_average": movie_data.get("vote_average"),
            "vote_count": movie_data.get("vote_count"),
            "popularity": movie_data.get("popularity"),
            "budget": movie_data.get("budget"),
            "revenue": movie_data.get("revenue"),
            "genres": [genre["name"] for genre in movie_data.get("genres", [])],
            "director": self._extract_director(credits_data),
            "cast": self._extract_main_cast(credits_data),
            "crew": self._extract_crew(credits_data),
            "production_companies": [comp["name"] for comp in movie_data.get("production_companies", [])],
            "production_countries": [country["name"] for country in movie_data.get("production_countries", [])],
            "spoken_languages": [lang["english_name"] for lang in movie_data.get("spoken_languages", [])],
            "keywords": [kw["name"] for kw in keywords_data.get("keywords", [])],
            "poster_url": f"{TMDB_IMAGE_BASE_URL}{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
            "backdrop_url": f"{TMDB_IMAGE_BASE_URL}{movie_data.get('backdrop_path')}" if movie_data.get('backdrop_path') else None
        }
        
        # Store in database
        await movie_db.add_or_update_movie(enhanced_movie)
        
        return enhanced_movie
    
    async def get_mood_based_recommendations(self, query: str, username: str = "demo_user", limit: int = 10) -> List[Dict]:
        """
        COMPLETELY SAFE mood-based search that avoids JSON parsing in database
        """
        # Analyze the query (keep existing mood analyzer)
        analysis = self.mood_analyzer.analyze_query(query)
        
        user = await self.get_user_or_create(username)
        user_id = user['user_id']
        
        async with movie_db.pool.acquire() as conn:
            # Get user's viewing history to avoid repetition
            user_history = await conn.fetch("""
                SELECT movie_id FROM user_ratings WHERE user_id = $1
            """, user_id)
            viewed_movie_ids = [r['movie_id'] for r in user_history]
            
            # Build base conditions (NO JSON PARSING)
            base_conditions = ["m.vote_average >= 6.0"]
            params = [user_id]
            param_count = 2
            
            # Genre matching using simple text search (SAFE)
            if analysis['genres']:
                genre_conditions = []
                for genre in analysis['genres']:
                    # Use simple text search instead of JSON parsing
                    genre_conditions.append(f"COALESCE(m.genres, '') ILIKE ${param_count}")
                    params.append(f'%{genre}%')
                    param_count += 1
                # Use OR instead of requiring all genres
                base_conditions.append(f"({' OR '.join(genre_conditions)})")
            
            # Runtime preferences
            if 'max_runtime' in analysis['preferences']:
                base_conditions.append(f"(m.runtime <= ${param_count} OR m.runtime IS NULL)")
                params.append(analysis['preferences']['max_runtime'])
                param_count += 1
            
            if 'min_runtime' in analysis['preferences']:
                base_conditions.append(f"(m.runtime >= ${param_count} OR m.runtime IS NULL)")
                params.append(analysis['preferences']['min_runtime'])
                param_count += 1
            
            # Year preferences
            if 'year_range' in analysis['preferences']:
                year_start, year_end = analysis['preferences']['year_range']
                base_conditions.append(f"(EXTRACT(YEAR FROM m.release_date) BETWEEN ${param_count} AND ${param_count + 1} OR m.release_date IS NULL)")
                params.extend([year_start, year_end])
                param_count += 2
            
            # Exclude viewed movies
            if viewed_movie_ids:
                base_conditions.append(f"m.movie_id != ALL(${param_count}::int[])")
                params.append(viewed_movie_ids)
                param_count += 1
            
            where_clause = " AND ".join(base_conditions)
            
            # Main query with diversity scoring (NO JSON FUNCTIONS)
            movies = await conn.fetch(f"""
                WITH mood_movies AS (
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.vote_average,
                        m.genres,
                        m.release_date,
                        m.overview,
                        m.runtime,
                        COALESCE(m.popularity, 50) as popularity,
                        COALESCE(ur.avg_rating, m.vote_average/2) as estimated_rating,
                        COALESCE(ur.rating_count, 0) as rating_count,
                        -- Diversity score: combines rating, popularity, and randomness
                        (m.vote_average * 0.4 + 
                        COALESCE(ur.avg_rating, 0) * 0.3 + 
                        LEAST(COALESCE(m.popularity, 50)/20, 5) * 0.2 +
                        RANDOM() * 0.1) as mood_score
                    FROM movies m
                    LEFT JOIN (
                        SELECT 
                            movie_id,
                            AVG(rating) as avg_rating,
                            COUNT(*) as rating_count
                        FROM user_ratings 
                        GROUP BY movie_id
                    ) ur ON m.movie_id = ur.movie_id
                    WHERE {where_clause}
                )
                SELECT * FROM mood_movies
                ORDER BY mood_score DESC
                LIMIT ${param_count}
            """, *params, limit * 2)  # Get more for diversity
            
            # Process results safely in Python (not in database)
            results = []
            for movie in movies:
                movie_dict = dict(movie)
                
                # Parse genres SAFELY in Python
                try:
                    if movie['genres'] and isinstance(movie['genres'], str):
                        genres_data = json.loads(movie['genres'])
                        if isinstance(genres_data, list):
                            movie_dict['genre_names'] = [g.get('name', 'Unknown') for g in genres_data if isinstance(g, dict)]
                        else:
                            movie_dict['genre_names'] = []
                    else:
                        movie_dict['genre_names'] = []
                except (json.JSONDecodeError, TypeError, AttributeError):
                    movie_dict['genre_names'] = ['Unknown']
                
                # Add mood match info
                movie_dict['mood_match'] = {
                    "matched_moods": analysis['moods'],
                    "matched_genres": [g for g in analysis['genres'] 
                                    if any(g.lower() in gn.lower() for gn in movie_dict['genre_names'])]
                }
                
                results.append(movie_dict)
                
                if len(results) >= limit:
                    break
            
            return results
    
    async def add_rating_with_sentiment(self, username: str, movie_id: int, 
                                      rating: float, review: str = None) -> Dict:
        """Add rating with sentiment analysis"""
        user = await self.get_user_or_create(username)
        
        # Analyze sentiment if review provided
        sentiment_data = None
        if review:
            sentiment_data = self.sentiment_analyzer.analyze_review(review)
        
        # Add rating to database
        success = await movie_db.add_user_rating(
            user['user_id'], movie_id, rating, review
        )
        
        if success and sentiment_data:
            # Store sentiment analysis (you might want to add a sentiment table)
            async with movie_db.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_activity_log (user_id, activity_type, movie_id, metadata)
                    VALUES ($1, 'sentiment_analysis', $2, $3)
                """, user['user_id'], movie_id, json.dumps(sentiment_data))
        
        return {
            "success": success,
            "sentiment_analysis": sentiment_data
        }
    
    def _extract_director(self, credits_data: Dict[str, Any]) -> Optional[str]:
        """Extract director from credits data."""
        crew = credits_data.get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                return person.get("name")
        return None
    
    def _extract_main_cast(self, credits_data: Dict[str, Any], limit: int = 10) -> List[str]:
        """Extract main cast members from credits data."""
        cast = credits_data.get("cast", [])
        return [person.get("name") for person in cast[:limit] if person.get("name")]
    
    def _extract_crew(self, credits_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key crew members."""
        crew = credits_data.get("crew", [])
        key_jobs = ["Director", "Producer", "Executive Producer", "Screenplay", "Writer", "Cinematography", "Music"]
        
        key_crew = []
        for person in crew:
            if person.get("job") in key_jobs:
                key_crew.append({
                    "name": person.get("name"),
                    "job": person.get("job")
                })
        
        return key_crew[:15]  # Limit to top 15 crew members
    
    async def get_user_or_create(self, username: str) -> Dict:
        """Get user or create if doesn't exist."""
        user = await movie_db.get_user(username=username)
        if not user:
            user = await movie_db.create_user(username)
        return user
    
    async def get_personalized_recommendations(self, username: str, limit: int = 10) -> List[Dict]:
        """Get personalized recommendations using ML algorithms."""
        user = await self.get_user_or_create(username)
        user_id = user['user_id']
        
        # Try collaborative filtering first
        collab_recs = await movie_db.get_collaborative_recommendations(user_id, limit // 2)
        
        # Get content-based recommendations
        content_recs = await movie_db.get_content_based_recommendations(user_id, limit // 2)
        
        # Combine and deduplicate
        all_recs = collab_recs + content_recs
        seen_movies = set()
        final_recs = []
        
        for rec in all_recs:
            if rec['movie_id'] not in seen_movies and len(final_recs) < limit:
                seen_movies.add(rec['movie_id'])
                final_recs.append(rec)
        
        # Fill with trending if needed
        if len(final_recs) < limit:
            trending = await movie_db.get_trending_movies(limit - len(final_recs))
            for movie in trending:
                if movie['movie_id'] not in seen_movies and len(final_recs) < limit:
                    final_recs.append(movie)
        
        return final_recs
    
    async def analyze_user_preferences(self, username: str) -> Dict:
        """Analyze user preferences and provide insights."""
        user = await self.get_user_or_create(username)
        stats = await movie_db.get_user_statistics(user['user_id'])
        
        # Create analysis summary
        analysis = {
            "user_profile": {
                "username": username,
                "total_ratings": stats['basic_stats'].get('total_ratings', 0),
                "average_rating": round(stats['basic_stats'].get('avg_rating', 0), 2),
                "favorites_count": stats['basic_stats'].get('favorites_count', 0)
            },
            "genre_preferences": stats['genre_preferences'][:5],
            "recent_activity": stats['recent_activity'],
            "recommendation_confidence": "high" if stats['basic_stats'].get('total_ratings', 0) > 10 else "medium" if stats['basic_stats'].get('total_ratings', 0) > 3 else "low"
        }
        
        return analysis

    async def get_enhanced_recommendations_v2(self, username: str = "demo_user", limit: int = 10) -> List[Dict]:
        """
        Enhanced recommendation system that provides more diverse results
        """
        user = await self.get_user_or_create(username)
        user_id = user['user_id']
        
        async with movie_db.pool.acquire() as conn:
            # Get user's rating history to avoid repetition
            user_rated_movies = await conn.fetch("""
                SELECT movie_id FROM user_ratings WHERE user_id = $1
            """, user_id)
            
            rated_movie_ids = [r['movie_id'] for r in user_rated_movies]
            
            # Strategy 1: Collaborative Filtering (if user has ratings)
            collaborative_movies = []
            if rated_movie_ids:
                collaborative_movies = await conn.fetch("""
                    WITH similar_users AS (
                        -- Find users with similar tastes
                        SELECT 
                            ur2.user_id,
                            COUNT(*) as common_movies,
                            AVG(ABS(ur1.rating - ur2.rating)) as rating_diff
                        FROM user_ratings ur1
                        JOIN user_ratings ur2 ON ur1.movie_id = ur2.movie_id
                        WHERE ur1.user_id = $1 AND ur2.user_id != $1
                        GROUP BY ur2.user_id
                        HAVING COUNT(*) >= 2
                        ORDER BY rating_diff ASC, common_movies DESC
                        LIMIT 10
                    ),
                    recommended_movies AS (
                        -- Get highly rated movies from similar users
                        SELECT 
                            m.movie_id,
                            m.title,
                            m.vote_average,
                            m.genres,
                            m.release_date,
                            m.overview,
                            AVG(ur.rating) as predicted_rating,
                            COUNT(ur.rating) as recommendation_strength,
                            'collaborative' as source
                        FROM similar_users su
                        JOIN user_ratings ur ON su.user_id = ur.user_id
                        JOIN movies m ON ur.movie_id = m.movie_id
                        WHERE ur.rating >= 4.0
                        AND ur.movie_id != ALL($2::int[])
                        GROUP BY m.movie_id, m.title, m.vote_average, m.genres, m.release_date, m.overview
                        HAVING COUNT(ur.rating) >= 2
                    )
                    SELECT * FROM recommended_movies
                    ORDER BY predicted_rating DESC, recommendation_strength DESC
                    LIMIT $3
                """, user_id, rated_movie_ids, limit // 2)
            
            # Strategy 2: High-quality unrated movies (discovery)
            remaining_slots = limit - len(collaborative_movies)
            if remaining_slots > 0:
                popular_unrated = await conn.fetch("""
                    SELECT DISTINCT
                        m.movie_id,
                        m.title,
                        m.vote_average,
                        m.genres,
                        m.release_date,
                        m.overview,
                        (m.vote_average / 2 + LEAST(m.popularity, 100) / 100 * 0.5) as predicted_rating,
                        'discovery' as source
                    FROM movies m
                    LEFT JOIN user_ratings ur ON m.movie_id = ur.movie_id AND ur.user_id = $1
                    WHERE ur.movie_id IS NULL
                    AND m.vote_average >= 6.5
                    AND m.movie_id != ALL($2::int[])
                    ORDER BY predicted_rating DESC
                    LIMIT $3
                """, user_id, rated_movie_ids, remaining_slots)
                
                # Combine strategies
                all_recommendations = list(collaborative_movies) + list(popular_unrated)
            else:
                all_recommendations = list(collaborative_movies)
            
            # Remove duplicates and limit results
            seen_movies = set()
            final_recommendations = []
            
            for movie in all_recommendations:
                if movie['movie_id'] not in seen_movies and len(final_recommendations) < limit:
                    seen_movies.add(movie['movie_id'])
                    final_recommendations.append(dict(movie))
            
            # If still not enough, add some random high-quality movies
            if len(final_recommendations) < limit:
                fallback_movies = await conn.fetch("""
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.vote_average,
                        m.genres,
                        m.release_date,
                        m.overview,
                        m.vote_average / 2 as predicted_rating,
                        'fallback' as source
                    FROM movies m
                    WHERE m.vote_average >= 7.5
                    AND m.movie_id != ALL($1::int[])
                    AND m.movie_id != ALL($2::int[])
                    ORDER BY RANDOM()
                    LIMIT $3
                """, rated_movie_ids, list(seen_movies), limit - len(final_recommendations))
                
                final_recommendations.extend([dict(movie) for movie in fallback_movies])
            
            return final_recommendations[:limit]

    async def get_diverse_mood_recommendations_v2(self, query: str, username: str = "demo_user", limit: int = 10) -> List[Dict]:
        """
        Enhanced mood-based search with better diversity
        """
        # Analyze the query (reuse existing mood analyzer)
        analysis = self.mood_analyzer.analyze_query(query)
        
        user = await self.get_user_or_create(username)
        user_id = user['user_id']
        
        async with movie_db.pool.acquire() as conn:
            # Get user's viewing history to avoid repetition
            user_history = await conn.fetch("""
                SELECT movie_id FROM user_ratings WHERE user_id = $1
            """, user_id)
            viewed_movie_ids = [r['movie_id'] for r in user_history]
            
            # Build a more flexible query
            base_conditions = ["m.vote_average >= 6.0"]  # Lower threshold
            params = [user_id]
            param_count = 2
            
            # Genre matching (more flexible)
            if analysis['genres']:
                genre_conditions = []
                for genre in analysis['genres']:
                    genre_conditions.append(f"COALESCE(m.genres, '') ILIKE ${param_count}")
                    params.append(f'%{genre}%')
                    param_count += 1
                # Use OR instead of requiring all genres
                base_conditions.append(f"({' OR '.join(genre_conditions)})")
            
            # Runtime preferences (more flexible ranges)
            if 'max_runtime' in analysis['preferences']:
                base_conditions.append(f"(m.runtime <= ${param_count} OR m.runtime IS NULL)")
                params.append(analysis['preferences']['max_runtime'])
                param_count += 1
            
            # Exclude already viewed movies
            if viewed_movie_ids:
                base_conditions.append(f"m.movie_id != ALL(${param_count}::int[])")
                params.append(viewed_movie_ids)
                param_count += 1
            
            where_clause = " AND ".join(base_conditions)
            
            # Main query with diversity scoring
            movies = await conn.fetch(f"""
                WITH mood_movies AS (
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.vote_average,
                        m.genres,
                        m.release_date,
                        m.overview,
                        m.runtime,
                        COALESCE(m.popularity, 50) as popularity,
                        COALESCE(ur.avg_rating, m.vote_average/2) as estimated_rating,
                        COALESCE(ur.rating_count, 0) as rating_count,
                        -- Diversity score: combines rating, popularity, and randomness
                        (m.vote_average * 0.4 + 
                        COALESCE(ur.avg_rating, 0) * 0.3 + 
                        LEAST(COALESCE(m.popularity, 50)/20, 5) * 0.2 +
                        RANDOM() * 0.1) as mood_score
                    FROM movies m
                    LEFT JOIN (
                        SELECT 
                            movie_id,
                            AVG(rating) as avg_rating,
                            COUNT(*) as rating_count
                        FROM user_ratings 
                        GROUP BY movie_id
                    ) ur ON m.movie_id = ur.movie_id
                    WHERE {where_clause}
                )
                SELECT * FROM mood_movies
                ORDER BY mood_score DESC
                LIMIT ${param_count}
            """, *params, limit * 2)  # Get more results to ensure diversity
            
            # Return diverse results
            results = [dict(movie) for movie in movies[:limit]]
            return results


class MoodAnalyzer:
    """Analyzes user mood and intent from natural language queries"""
    
    def __init__(self):
        # Mood to genre mappings
        self.mood_genre_map = {
            "funny": ["Comedy"],
            "light-hearted": ["Comedy", "Animation", "Family"],
            "lighthearted": ["Comedy", "Animation", "Family"],
            "uplifting": ["Comedy", "Family", "Music", "Animation"],
            "happy": ["Comedy", "Family", "Animation", "Music"],
            "sad": ["Drama", "Romance"],
            "exciting": ["Action", "Adventure", "Thriller"],
            "thrilling": ["Thriller", "Action", "Mystery"],
            "scary": ["Horror", "Thriller"],
            "romantic": ["Romance", "Drama"],
            "thoughtful": ["Drama", "Documentary", "Biography"],
            "intellectual": ["Documentary", "Biography", "Drama", "Science Fiction"],
            "adventurous": ["Adventure", "Action", "Fantasy"],
            "mysterious": ["Mystery", "Thriller", "Crime"],
            "dark": ["Horror", "Thriller", "Crime", "Mystery"],
            "inspiring": ["Biography", "Drama", "Documentary", "Sport"],
            "relaxing": ["Family", "Animation", "Comedy"],
            "nostalgic": ["Family", "Animation", "Comedy", "Drama"]
        }
        
        # Context keywords
        self.context_keywords = {
            "tonight": {"time_preference": "evening"},
            "today": {"time_preference": "now"},
            "weekend": {"time_preference": "weekend"},
            "date night": {"context": "romantic", "viewer_count": 2},
            "with kids": {"context": "family", "rating_limit": "PG-13"},
            "family": {"context": "family", "rating_limit": "PG-13"},
            "alone": {"viewer_count": 1},
            "friends": {"context": "group", "viewer_count": 3},
            "party": {"context": "group", "viewer_count": 5}
        }
        
        # Duration preferences
        self.duration_keywords = {
            "short": {"max_runtime": 90},
            "quick": {"max_runtime": 90},
            "long": {"min_runtime": 120},
            "epic": {"min_runtime": 150}
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Extract mood, context, and preferences from natural language query"""
        query_lower = query.lower()
        
        # Initialize analysis result
        analysis = {
            "moods": [],
            "genres": [],
            "context": {},
            "preferences": {},
            "keywords": []
        }
        
        # Extract moods and corresponding genres
        for mood, genres in self.mood_genre_map.items():
            if mood in query_lower:
                analysis["moods"].append(mood)
                analysis["genres"].extend(genres)
        
        # Extract context
        for keyword, context in self.context_keywords.items():
            if keyword in query_lower:
                analysis["context"].update(context)
        
        # Extract duration preferences
        for keyword, pref in self.duration_keywords.items():
            if keyword in query_lower:
                analysis["preferences"].update(pref)
        
        # Extract specific movie characteristics
        if "new" in query_lower or "latest" in query_lower:
            analysis["preferences"]["release_year_min"] = datetime.now().year - 2
        if "classic" in query_lower:
            analysis["preferences"]["release_year_max"] = 2000
        if "highly rated" in query_lower or "best" in query_lower:
            analysis["preferences"]["min_rating"] = 7.5
        
        # Remove duplicates from genres
        analysis["genres"] = list(set(analysis["genres"]))
        
        # Extract any remaining important words
        important_words = ["action", "drama", "comedy", "horror", "sci-fi", "fantasy", 
                          "animated", "documentary", "thriller", "mystery", "western"]
        for word in important_words:
            if word in query_lower and word not in analysis["keywords"]:
                analysis["keywords"].append(word)
        
        return analysis
    
    

class SentimentAnalyzer:
    """Analyzes sentiment from movie reviews"""
    
    def analyze_review(self, review_text: str) -> Dict:
        """Analyze sentiment and extract key aspects from review"""
        if not review_text:
            return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0}
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(review_text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.3:
            sentiment = "positive"
        elif polarity < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract aspects mentioned in review
        aspects = []
        aspect_keywords = {
            "acting": ["acting", "actor", "actress", "performance", "cast"],
            "story": ["story", "plot", "storyline", "narrative", "script"],
            "direction": ["direction", "director", "directed", "filmmaking"],
            "visuals": ["visual", "cinematography", "effects", "cgi", "animation"],
            "music": ["music", "score", "soundtrack", "song"],
            "pacing": ["pacing", "pace", "slow", "fast", "boring", "engaging"],
            "emotion": ["emotional", "moving", "touching", "feel", "heart"]
        }
        
        review_lower = review_text.lower()
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in review_lower for keyword in keywords):
                aspects.append(aspect)
        
        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "aspects": aspects,
            "is_recommendation": polarity > 0.5  # Would they recommend it?
        }
    
    async def analyze_movie_sentiment(self, movie_id: int, db_connection) -> Dict:
        """Analyze overall sentiment for a movie based on all reviews"""
        # Get all reviews for the movie
        reviews = await db_connection.fetch("""
            SELECT rating, review_text 
            FROM user_ratings 
            WHERE movie_id = $1 AND review_text IS NOT NULL
        """, movie_id)
        
        if not reviews:
            return {"overall_sentiment": "unknown", "review_count": 0}
        
        sentiments = []
        total_polarity = 0
        aspects_count = {}
        
        for review in reviews:
            analysis = self.analyze_review(review['review_text'])
            sentiments.append(analysis['sentiment'])
            total_polarity += analysis['polarity']
            
            for aspect in analysis['aspects']:
                aspects_count[aspect] = aspects_count.get(aspect, 0) + 1
        
        # Calculate overall sentiment
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        
        if positive_count > negative_count * 2:
            overall = "very_positive"
        elif positive_count > negative_count:
            overall = "positive"
        elif negative_count > positive_count * 2:
            overall = "very_negative"
        elif negative_count > positive_count:
            overall = "negative"
        else:
            overall = "mixed"
        
        return {
            "overall_sentiment": overall,
            "average_polarity": round(total_polarity / len(reviews), 3),
            "review_count": len(reviews),
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": sentiments.count('neutral')
            },
            "top_aspects": sorted(aspects_count.items(), key=lambda x: x[1], reverse=True)[:3]
        }

# Initialize server instance
movie_server = EnhancedMovieDiscoveryServer()


@app.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools including ML and NLP features."""
    return [
        types.Tool(
            name="search_movies",
            description="Search for movies and add them to the ML database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Movie title or keywords to search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_personalized_recommendations",
            description="Get AI-powered personalized movie recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username for personalized recommendations",
                        "default": "demo_user"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations (default: 10)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="rate_movie",
            description="Rate a movie and update user preferences with optional sentiment analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username",
                        "default": "demo_user"
                    },
                    "movie_id": {
                        "type": "integer",
                        "description": "TMDB movie ID"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Rating from 0.5 to 5.0",
                        "minimum": 0.5,
                        "maximum": 5.0
                    },
                    "review": {
                        "type": "string",
                        "description": "Optional review text (will be analyzed for sentiment)"
                    }
                },
                "required": ["movie_id", "rating"]
            }
        ),
        types.Tool(
            name="analyze_user_preferences",
            description="Analyze user's movie preferences and viewing patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to analyze",
                        "default": "demo_user"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_trending_movies",
            description="Get currently trending movies with ML-powered insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_period": {
                        "type": "string",
                        "enum": ["day", "week"],
                        "description": "Time period for trending analysis",
                        "default": "week"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="find_similar_movies",
            description="Find movies similar to a given movie using ML algorithms",
            inputSchema={
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "TMDB movie ID to find similar movies for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of similar movies to return",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": ["movie_id"]
            }
        ),
        types.Tool(
            name="mood_based_search",
            description="Get movie recommendations based on mood or natural language query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing mood or preferences (e.g., 'funny movie for tonight', 'something uplifting', 'scary but not too intense')"
                    },
                    "username": {
                        "type": "string",
                        "description": "Username for personalization",
                        "default": "demo_user"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="analyze_movie_sentiment",
            description="Analyze overall sentiment for a movie based on user reviews",
            inputSchema={
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "TMDB movie ID to analyze"
                    }
                },
                "required": ["movie_id"]
            }
        ),
        types.Tool(
            name="analyze_user_preferences",
            description="Analyze a user's movie preferences and viewing patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to analyze preferences for"
                    }
                },
                "required": ["username"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls with JSON-safe database operations."""
    try:
        # ALSO REPLACE the search_movies section in your @app.call_tool() function:

        if name == "search_movies":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            
            if not query:
                return [types.TextContent(type="text", text="Please provide a search query.")]
            
            # Try TMDB search first for better results
            try:
                data = await movie_server.fetch_tmdb_data("search/movie", {"query": query})
                tmdb_movies = data.get("results", [])[:limit]
                
                if tmdb_movies:
                    result = f"🎬 Found {len(tmdb_movies)} movies for '{query}':\n\n"
                    for i, movie in enumerate(tmdb_movies, 1):
                        # Store movie in database for future use
                        try:
                            await movie_server.get_enhanced_movie_details(movie['id'])
                        except:
                            pass
                        
                        release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                        result += f"{i}. **{movie['title']}** ({release_year})\n"
                        result += f"   ⭐ Rating: {movie['vote_average']}/10 | 🔥 Popularity: {movie['popularity']:.1f}\n"
                        result += f"   📝 {movie.get('overview', 'No overview available.')[:150]}...\n"
                        result += f"   🆔 ID: {movie['id']}\n\n"
                    
                    return [types.TextContent(type="text", text=result)]
            except Exception as e:
                # Fall back to database search if TMDB fails
                pass
            
            # Database search as fallback
            async with movie_db.pool.acquire() as conn:
                movies = await conn.fetch("""
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.vote_average,
                        m.release_date,
                        m.overview,
                        m.runtime,
                        CASE 
                            WHEN LOWER(m.title) LIKE LOWER($2) THEN 100
                            WHEN LOWER(m.title) LIKE LOWER($1) THEN 90
                            WHEN LOWER(COALESCE(m.overview, '')) LIKE LOWER($1) THEN 80
                            ELSE 70
                        END as relevance_score
                    FROM movies m
                    WHERE (
                        LOWER(m.title) LIKE LOWER($1)
                        OR LOWER(COALESCE(m.overview, '')) LIKE LOWER($1)
                    )
                    AND m.vote_average >= 5.0
                    ORDER BY 
                        relevance_score DESC,
                        m.vote_average DESC,
                        COALESCE(m.popularity, 0) DESC
                    LIMIT $3
                """, f'%{query}%', f'{query}%', limit)
            
            if not movies:
                result = f"🎬 No movies found for '{query}'\n\n"
                result += "Try different keywords or check your database."
            else:
                result = f"🎬 Found {len(movies)} movies for '{query}':\n\n"
                for i, movie in enumerate(movies, 1):
                    release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                    result += f"{i}. **{movie['title']}** ({release_year})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                    if movie.get('overview'):
                        result += f"   📝 {movie['overview'][:150]}...\n"
                    result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "analyze_user_preferences":
            username = arguments.get("username")
            
            if not username:
                return [types.TextContent(type="text", text="Please provide a username to analyze.")]
            
            # Get user
            user = await movie_server.get_user_or_create(username)
            user_id = user['user_id']
            
            async with movie_db.pool.acquire() as conn:
                # Basic stats
                basic_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_ratings,
                        AVG(rating) as avg_rating,
                        COUNT(CASE WHEN review_text IS NOT NULL THEN 1 END) as reviews_written,
                        COUNT(CASE WHEN rating >= 4.5 THEN 1 END) as favorites_count
                    FROM user_ratings 
                    WHERE user_id = $1
                """, user_id)
                
                # Genre preferences
                genre_prefs = await conn.fetch("""
                    SELECT 
                        genre_name,
                        COUNT(*) as movie_count,
                        AVG(rating) as avg_rating,
                        COUNT(CASE WHEN rating >= 4.0 THEN 1 END) as high_ratings
                    FROM (
                        SELECT 
                            ur.rating,
                            jsonb_array_elements(m.genres::jsonb)->>'name' as genre_name
                        FROM user_ratings ur
                        JOIN movies m ON ur.movie_id = m.movie_id
                        WHERE ur.user_id = $1
                        AND m.genres IS NOT NULL
                        AND m.genres != 'null'
                    ) genre_ratings
                    GROUP BY genre_name
                    HAVING COUNT(*) >= 2
                    ORDER BY AVG(rating) DESC, COUNT(*) DESC
                """, user_id)
                
                # Recent activity
                recent_ratings = await conn.fetch("""
                    SELECT m.title, ur.rating, ur.created_at
                    FROM user_ratings ur
                    JOIN movies m ON ur.movie_id = m.movie_id
                    WHERE ur.user_id = $1
                    ORDER BY ur.created_at DESC
                    LIMIT 5
                """, user_id)
                
                # Rating distribution
                rating_dist = await conn.fetch("""
                    SELECT rating, COUNT(*) as count
                    FROM user_ratings 
                    WHERE user_id = $1
                    GROUP BY rating
                    ORDER BY rating DESC
                """, user_id)
            
            # Format results
            if basic_stats['total_ratings'] == 0:
                result = f"📊 **{username}'s Movie Preferences**\n\n"
                result += "No ratings found! Start rating some movies to see your preferences."
            else:
                result = f"📊 **{username}'s Movie Preferences Analysis**\n\n"
                
                # Basic stats
                result += f"**📈 Rating Overview:**\n"
                result += f"• Total movies rated: {basic_stats['total_ratings']}\n"
                result += f"• Average rating: {float(basic_stats['avg_rating']):.1f}/5.0\n"
                result += f"• Reviews written: {basic_stats['reviews_written']}\n"
                result += f"• Favorites (4.5+ stars): {basic_stats['favorites_count']}\n\n"
                
                # Genre preferences
                if genre_prefs:
                    result += f"**🎭 Favorite Genres:**\n"
                    for i, genre in enumerate(genre_prefs[:5], 1):
                        result += f"{i}. {genre['genre_name']}: {float(genre['avg_rating']):.1f}/5 ({genre['movie_count']} movies)\n"
                    result += "\n"
                
                # Rating patterns
                if rating_dist:
                    result += f"**⭐ Rating Distribution:**\n"
                    for rating in rating_dist:
                        stars = "⭐" * int(rating['rating'])
                        result += f"{stars} ({rating['rating']}): {rating['count']} movies\n"
                    result += "\n"
                
                # Recent activity
                if recent_ratings:
                    result += f"**🕒 Recent Activity:**\n"
                    for rating in recent_ratings:
                        result += f"• {rating['title']}: {rating['rating']}/5\n"
                
                # Personality insights
                avg_rating = float(basic_stats['avg_rating'])
                if avg_rating >= 4.0:
                    result += f"\n**🎯 Viewing Style:** Enthusiastic viewer - you love most movies!\n"
                elif avg_rating >= 3.5:
                    result += f"\n**🎯 Viewing Style:** Balanced critic - you appreciate good cinema.\n"
                elif avg_rating >= 3.0:
                    result += f"\n**🎯 Viewing Style:** Selective viewer - you have high standards.\n"
                else:
                    result += f"\n**🎯 Viewing Style:** Tough critic - very discerning taste!\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "get_personalized_recommendations":
            username = arguments.get("username")
            limit = arguments.get("limit", 10)
            
            if not username:
                return [types.TextContent(type="text", text="Please provide a username for personalized recommendations.")]
            
            # Get user safely
            user = await movie_server.get_user_or_create(username)
            user_id = user['user_id']
            
            # Use safe collaborative filtering
            async with movie_db.pool.acquire() as conn:
                # Get user's rated movies
                user_movies = await conn.fetch("""
                    SELECT movie_id FROM user_ratings WHERE user_id = $1
                """, user_id)
                
                rated_movie_ids = [r['movie_id'] for r in user_movies]
                
                if not rated_movie_ids:
                    # Return popular movies if no ratings
                    recommendations = await conn.fetch("""
                        SELECT 
                            m.movie_id,
                            m.title,
                            m.vote_average,
                            m.release_date,
                            m.overview,
                            m.vote_average / 2 as predicted_rating
                        FROM movies m
                        WHERE m.vote_average >= 7.5
                        ORDER BY m.vote_average DESC
                        LIMIT $1
                    """, limit)
                else:
                    # Find similar users and get recommendations
                    recommendations = await conn.fetch("""
                        WITH similar_users AS (
                            SELECT 
                                ur2.user_id,
                                COUNT(*) as common_movies,
                                AVG(ABS(ur1.rating - ur2.rating)) as rating_diff
                            FROM user_ratings ur1
                            JOIN user_ratings ur2 ON ur1.movie_id = ur2.movie_id
                            WHERE ur1.user_id = $1 AND ur2.user_id != $1
                            GROUP BY ur2.user_id
                            HAVING COUNT(*) >= 2
                            ORDER BY rating_diff ASC, common_movies DESC
                            LIMIT 10
                        )
                        SELECT 
                            m.movie_id,
                            m.title,
                            m.vote_average,
                            m.release_date,
                            m.overview,
                            AVG(ur.rating) as predicted_rating
                        FROM similar_users su
                        JOIN user_ratings ur ON su.user_id = ur.user_id
                        JOIN movies m ON ur.movie_id = m.movie_id
                        WHERE ur.rating >= 4.0
                        AND ur.movie_id != ALL($2::int[])
                        GROUP BY m.movie_id, m.title, m.vote_average, m.release_date, m.overview
                        ORDER BY AVG(ur.rating) DESC
                        LIMIT $3
                    """, user_id, rated_movie_ids, limit)
            
            if not recommendations:
                result = f"🤖 **No personalized recommendations found for {username}**\n\n"
                result += "Rate some movies first to get better recommendations!"
            else:
                result = f"🤖 **AI-Powered Recommendations for {username}**\n\n"
                for i, movie in enumerate(recommendations, 1):
                    release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                    result += f"{i}. **{movie['title']}** ({release_year})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                    if 'predicted_rating' in movie:
                        result += f"   🎯 Predicted for you: {movie['predicted_rating']:.1f}/5.0\n"
                    result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "rate_movie":
            username = arguments.get("username")
            movie_id = arguments.get("movie_id")
            rating = arguments.get("rating")
            review = arguments.get("review")
            
            if not username:
                return [types.TextContent(type="text", text="Please provide a username.")]
            if not movie_id:
                return [types.TextContent(type="text", text="Please provide a movie ID.")]
            if not rating:
                return [types.TextContent(type="text", text="Please provide a rating (1-5).")]
            
            # Get user
            user = await movie_server.get_user_or_create(username)
            
            # Get movie details
            try:
                movie = await movie_server.get_enhanced_movie_details(movie_id)
                movie_title = movie['title']
            except:
                movie_title = f"Movie ID {movie_id}"
            
            # Add rating with sentiment analysis if review provided
            sentiment_data = None
            if review:
                sentiment_data = movie_server.sentiment_analyzer.analyze_review(review)
            
            # Add rating
            success = await movie_db.add_user_rating(
                user['user_id'], movie_id, rating, review
            )
            
            if success:
                result = f"✅ **Rating Added Successfully!**\n\n"
                result += f"👤 **User:** {username}\n"
                result += f"🎬 **Movie:** {movie_title}\n"
                result += f"⭐ **Rating:** {rating}/5.0\n"
                if review:
                    result += f"📝 **Review:** {review}\n"
                    if sentiment_data:
                        result += f"\n**📊 Sentiment Analysis:**\n"
                        result += f"• Sentiment: {sentiment_data['sentiment'].title()}\n"
                        result += f"• Polarity: {sentiment_data['polarity']} (-1 to 1 scale)\n"
                        if sentiment_data['aspects']:
                            result += f"• Discussed: {', '.join(sentiment_data['aspects'])}\n"
                result += f"\n🤖 Your recommendations will now be updated!"
            else:
                result = f"❌ Failed to add rating. Please try again."
            
            return [types.TextContent(type="text", text=result)]
        

        elif name == "mood_based_search":
            query = arguments.get("query")
            username = arguments.get("username", "anonymous")
            limit = arguments.get("limit", 10)
            
            if not query:
                return [types.TextContent(type="text", text="Please provide a mood or preference query.")]
            
            # Map query to actual genres
            query_lower = query.lower()
            target_genres = []
            
            if any(word in query_lower for word in ['funny', 'comedy', 'laugh', 'hilarious']):
                target_genres.append('Comedy')
            if any(word in query_lower for word in ['action', 'fight', 'adventure', 'explosion']):
                target_genres.append('Action')
            if any(word in query_lower for word in ['scary', 'horror', 'terror']):
                target_genres.append('Horror')
            if any(word in query_lower for word in ['romantic', 'love', 'romance']):
                target_genres.append('Romance')
            if any(word in query_lower for word in ['thriller', 'suspense', 'mystery']):
                target_genres.append('Thriller')
            
            # Get user safely
            user = await movie_server.get_user_or_create(username)
            user_id = user['user_id']
            
            async with movie_db.pool.acquire() as conn:
                # Get user's viewing history
                user_history = await conn.fetch("""
                    SELECT movie_id FROM user_ratings WHERE user_id = $1
                """, user_id)
                viewed_movie_ids = [r['movie_id'] for r in user_history]
                
                if target_genres:
                    # Build genre filter conditions
                    genre_conditions = []
                    for genre in target_genres:
                        genre_conditions.append(f"m.genres::text ILIKE '%\"{genre}\"%'")
                    
                    if len(target_genres) > 1:
                        # Multiple genres = require ALL (Action AND Comedy)
                        genre_filter = " AND ".join(genre_conditions)
                    else:
                        # Single genre = just that genre
                        genre_filter = " OR ".join(genre_conditions)
                    
                    # Build exclude condition
                    if viewed_movie_ids:
                        exclude_condition = "AND m.movie_id != ALL($1::int[])"
                        exclude_params = [viewed_movie_ids]
                    else:
                        exclude_condition = ""
                        exclude_params = []
                    
                    # Execute query with proper parameter handling
                    if viewed_movie_ids:
                        movies = await conn.fetch(f"""
                            SELECT 
                                m.movie_id,
                                m.title,
                                m.vote_average,
                                m.release_date,
                                m.overview,
                                m.genres
                            FROM movies m
                            WHERE ({genre_filter})
                            AND m.vote_average >= 6.5
                            AND m.genres IS NOT NULL
                            AND m.movie_id != ALL($1::int[])
                            ORDER BY m.vote_average DESC, RANDOM()
                            LIMIT $2
                        """, viewed_movie_ids, limit)
                    else:
                        movies = await conn.fetch(f"""
                            SELECT 
                                m.movie_id,
                                m.title,
                                m.vote_average,
                                m.release_date,
                                m.overview,
                                m.genres
                            FROM movies m
                            WHERE ({genre_filter})
                            AND m.vote_average >= 6.5
                            AND m.genres IS NOT NULL
                            ORDER BY m.vote_average DESC, RANDOM()
                            LIMIT $1
                        """, limit)
                else:
                    # No specific genres - return popular movies
                    if viewed_movie_ids:
                        movies = await conn.fetch("""
                            SELECT movie_id, title, vote_average, release_date, overview, genres
                            FROM movies 
                            WHERE vote_average >= 7.5 
                            AND movie_id != ALL($1::int[])
                            ORDER BY vote_average DESC 
                            LIMIT $2
                        """, viewed_movie_ids, limit)
                    else:
                        movies = await conn.fetch("""
                            SELECT movie_id, title, vote_average, release_date, overview, genres
                            FROM movies 
                            WHERE vote_average >= 7.5 
                            ORDER BY vote_average DESC 
                            LIMIT $1
                        """, limit)
            
            if not movies:
                result = f"🎭 **No movies found for mood: '{query}'**\n\n"
                result += f"Looking for: {', '.join(target_genres) if target_genres else 'Popular movies'}\n"
                result += "Try different keywords or add more movies to your database!"
            else:
                result = f"🎭 **Mood-Based Recommendations: '{query}'**\n\n"
                result += f"Found {len(movies)} movies with genres: {', '.join(target_genres) if target_genres else 'Popular movies'}\n\n"
                
                for i, movie in enumerate(movies, 1):
                    release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                    result += f"{i}. **{movie['title']}** ({release_year})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                    
                    # Show genres
                    try:
                        if movie['genres']:
                            import json
                            genres_data = json.loads(movie['genres'])
                            genre_names = [g['name'] for g in genres_data]
                            result += f"   🎭 Genres: {', '.join(genre_names)}\n"
                    except:
                        pass
                        
                    result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "analyze_movie_sentiment":
            movie_id = arguments.get("movie_id")
            
            if not movie_id:
                return [types.TextContent(type="text", text="Please provide a movie ID.")]
            
            # Get movie details
            try:
                movie = await movie_server.get_enhanced_movie_details(movie_id)
            except:
                return [types.TextContent(type="text", text="Movie not found.")]
            
            # Get sentiment analysis
            async with movie_db.pool.acquire() as conn:
                sentiment = await movie_server.sentiment_analyzer.analyze_movie_sentiment(movie_id, conn)
            
            # Format response
            result = f"😊 **Sentiment Analysis for '{movie['title']}'**\n\n"
            
            if sentiment['review_count'] == 0:
                result += "No reviews available for sentiment analysis.\n"
                result += "Users need to add reviews for this movie first!"
            else:
                result += f"📊 **Analysis Results:**\n"
                result += f"• Overall Sentiment: {sentiment['overall_sentiment'].replace('_', ' ').title()}\n"
                result += f"• Average Polarity: {sentiment['average_polarity']}\n"
                result += f"• Reviews Analyzed: {sentiment['review_count']}\n\n"
                
                if sentiment['sentiment_distribution']:
                    result += f"**Sentiment Breakdown:**\n"
                    for sentiment_type, count in sentiment['sentiment_distribution'].items():
                        result += f"• {sentiment_type.title()}: {count}\n"
                
                if sentiment['top_aspects']:
                    result += f"\n**Most Discussed Aspects:**\n"
                    for aspect, count in sentiment['top_aspects']:
                        result += f"• {aspect}: mentioned {count} times\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "get_trending_movies":
            time_period = arguments.get("time_period", "week")
            limit = arguments.get("limit", 20)
            
            # Get trending from TMDB
            try:
                data = await movie_server.fetch_tmdb_data(f"trending/movie/{time_period}")
                movies = data.get("results", [])[:limit]
                
                result = f"🔥 **Trending Movies This {time_period.title()}**\n\n"
                for i, movie in enumerate(movies, 1):
                    # Store movie for future use
                    try:
                        await movie_server.get_enhanced_movie_details(movie['id'])
                    except:
                        pass
                    
                    release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                    result += f"{i}. **{movie['title']}** ({release_year})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10 | 🔥 Popularity: {movie['popularity']:.1f}\n"
                    result += f"   🆔 ID: {movie['id']}\n\n"
                
                return [types.TextContent(type="text", text=result)]
            except Exception as e:
                return [types.TextContent(type="text", text=f"❌ Error getting trending movies: {str(e)}")]
        
        elif name == "find_similar_movies":
            movie_id = arguments.get("movie_id")
            limit = arguments.get("limit", 10)
            
            if not movie_id:
                return [types.TextContent(type="text", text="Please provide a movie ID.")]
            
            # Get the source movie details
            try:
                source_movie = await movie_server.get_enhanced_movie_details(movie_id)
            except:
                return [types.TextContent(type="text", text="Source movie not found.")]
            
            # Find similar movies using TMDB
            try:
                data = await movie_server.fetch_tmdb_data(f"movie/{movie_id}/similar")
                similar_movies = data.get("results", [])[:limit]
                
                result = f"🎭 **Movies Similar to '{source_movie['title']}'**\n\n"
                
                if not similar_movies:
                    result += "No similar movies found."
                else:
                    for i, movie in enumerate(similar_movies, 1):
                        release_year = str(movie.get('release_date', 'Unknown'))[:4] if movie.get('release_date') else 'Unknown'
                        result += f"{i}. **{movie['title']}** ({release_year})\n"
                        result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                        result += f"   🆔 ID: {movie['id']}\n\n"
                
                return [types.TextContent(type="text", text=result)]
            except Exception as e:
                return [types.TextContent(type="text", text=f"❌ Error finding similar movies: {str(e)}")]
        
        else:
            return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    
    except Exception as e:
        error_msg = f"❌ Error in {name}: {str(e)}"
        return [types.TextContent(type="text", text=error_msg)]


async def main():
    # Initialize the enhanced server
    await movie_server.initialize()
    
    try:
        # Setup the server connection
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="movie-discovery",
                    server_version="2.0.0",
                    capabilities=app.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await movie_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())