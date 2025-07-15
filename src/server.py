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
        """Get movie recommendations based on natural language mood query"""
        
        # Analyze the query
        analysis = self.mood_analyzer.analyze_query(query)
        
        # Get user for personalization
        user = await self.get_user_or_create(username)
        user_id = user['user_id']
        
        # Build dynamic SQL query based on analysis
        conditions = []
        params = []
        param_count = 2
        
        # Add genre conditions
        if analysis['genres']:
            genre_conditions = []
            for genre in analysis['genres']:
                genre_conditions.append(f"m.genres::text ILIKE ${param_count}")
                params.append(f'%"{genre}"%')
                param_count += 1
            conditions.append(f"({' OR '.join(genre_conditions)})")
        
        # Add runtime conditions
        if 'max_runtime' in analysis['preferences']:
            conditions.append(f"m.runtime <= ${param_count}")
            params.append(analysis['preferences']['max_runtime'])
            param_count += 1
        
        if 'min_runtime' in analysis['preferences']:
            conditions.append(f"m.runtime >= ${param_count}")
            params.append(analysis['preferences']['min_runtime'])
            param_count += 1
        
        # Add rating conditions
        min_rating = analysis['preferences'].get('min_rating', 6.0)
        conditions.append(f"m.vote_average >= ${param_count}")
        params.append(min_rating)
        param_count += 1
        
        # Add year conditions
        if 'release_year_min' in analysis['preferences']:
            conditions.append(f"EXTRACT(YEAR FROM m.release_date) >= ${param_count}")
            params.append(analysis['preferences']['release_year_min'])
            param_count += 1
        
        if 'release_year_max' in analysis['preferences']:
            conditions.append(f"EXTRACT(YEAR FROM m.release_date) <= ${param_count}")
            params.append(analysis['preferences']['release_year_max'])
            param_count += 1
        
        # Build final query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        async with movie_db.pool.acquire() as conn:
            # Get movies matching mood criteria
            movies = await conn.fetch(f"""
                WITH user_seen AS (
                    SELECT movie_id FROM user_ratings WHERE user_id = $1
                ),
                mood_movies AS (
                    SELECT 
                        m.*,
                        COALESCE(AVG(ur.rating), m.vote_average/2) as avg_user_rating,
                        COUNT(ur.rating) as rating_count
                    FROM movies m
                    LEFT JOIN user_ratings ur ON m.movie_id = ur.movie_id
                    WHERE {where_clause}
                        AND m.movie_id NOT IN (SELECT movie_id FROM user_seen)
                        AND m.vote_count >= 100
                    GROUP BY m.movie_id, m.title, m.release_date, m.overview, 
                             m.genres, m.vote_average, m.vote_count, m.popularity,
                             m.runtime, m.budget, m.revenue, m.director, m.cast_members,
                             m.crew_members, m.poster_url, m.backdrop_url,
                             m.production_companies, m.production_countries,
                             m.spoken_languages, m.keywords, m.created_at, m.updated_at
                )
                SELECT 
                    mm.*,
                    -- Mood score calculation
                    (mm.avg_user_rating * 0.4 + 
                     mm.vote_average/2 * 0.3 + 
                     LEAST(mm.popularity/100, 5) * 0.3) as mood_score
                FROM mood_movies mm
                ORDER BY mood_score DESC
                LIMIT ${param_count}
            """, user_id, *params, limit)
            
            # Add sentiment analysis to results
            results = []
            for movie in movies:
                movie_dict = dict(movie)
                
                # Add sentiment if reviews exist
                sentiment = await self.sentiment_analyzer.analyze_movie_sentiment(
                    movie['movie_id'], conn
                )
                movie_dict['sentiment_analysis'] = sentiment
                
                # Add mood match explanation
                movie_dict['mood_match'] = {
                    "matched_moods": analysis['moods'],
                    "matched_genres": [g for g in analysis['genres'] 
                                     if g in movie_dict.get('genres', '')]
                }
                
                results.append(movie_dict)
            
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
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls with enhanced ML features including NLP."""
    try:
        if name == "search_movies":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            
            # Search TMDB
            data = await movie_server.fetch_tmdb_data("search/movie", {"query": query})
            movies = data.get("results", [])[:limit]
            
            result = f"🎬 Found {len(movies)} movies for '{query}':\n\n"
            for i, movie in enumerate(movies, 1):
                # Store movie in database for future ML processing
                await movie_server.get_enhanced_movie_details(movie['id'])
                
                result += f"{i}. **{movie['title']}** ({movie.get('release_date', 'Unknown')[:4]})\n"
                result += f"   ⭐ Rating: {movie['vote_average']}/10 | 🔥 Popularity: {movie['popularity']:.1f}\n"
                result += f"   📝 {movie.get('overview', 'No overview available.')[:150]}...\n"
                result += f"   🆔 ID: {movie['id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "get_personalized_recommendations":
            username = arguments.get("username", "demo_user")
            limit = arguments.get("limit", 10)
            
            recommendations = await movie_server.get_personalized_recommendations(username, limit)
            
            if not recommendations:
                result = f"🤖 **No Personalized Recommendations Available**\n\n"
                result += "Rate some movies first to get personalized suggestions!\n\n"
                result += "Try using `rate_movie` to rate movies you've watched."
            else:
                result = f"🤖 **AI-Powered Recommendations for {username}**\n\n"
                for i, movie in enumerate(recommendations, 1):
                    result += f"{i}. **{movie['title']}** ({movie.get('release_date', 'Unknown')[:4]})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                    if 'predicted_rating' in movie:
                        result += f"   🎯 Predicted for you: {movie['predicted_rating']:.1f}/5.0\n"
                    result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "rate_movie":
            username = arguments.get("username", "demo_user")
            movie_id = arguments.get("movie_id")
            rating = arguments.get("rating")
            review = arguments.get("review")
            
            # Get user
            user = await movie_server.get_user_or_create(username)
            
            # Get movie details to show what was rated
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
                result += f"\n🤖 Your recommendations will now be updated based on this rating!"
            else:
                result = f"❌ Failed to add rating. Please try again."
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "analyze_user_preferences":
            username = arguments.get("username", "demo_user")
            
            analysis = await movie_server.analyze_user_preferences(username)
            
            result = f"📊 **User Analysis for {username}**\n\n"
            
            profile = analysis['user_profile']
            result += f"**📈 Profile Summary:**\n"
            result += f"• Total Ratings: {profile['total_ratings']}\n"
            result += f"• Average Rating: {profile['average_rating']}/5.0\n"
            result += f"• Favorites: {profile['favorites_count']}\n"
            result += f"• Recommendation Confidence: {analysis['recommendation_confidence'].title()}\n\n"
            
            if analysis['genre_preferences']:
                result += f"**🎭 Favorite Genres:**\n"
                for genre in analysis['genre_preferences']:
                    result += f"• {genre['genre']}: {genre['avg_rating']:.1f}/5.0 avg ({genre['count']} movies)\n"
                result += "\n"
            
            if analysis['recent_activity']:
                result += f"**📊 Recent Activity (30 days):**\n"
                for activity in analysis['recent_activity']:
                    result += f"• {activity['activity_type'].title()}: {activity['count']} times\n"
            else:
                result += f"**📊 Recent Activity:** No recent activity\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "get_trending_movies":
            time_period = arguments.get("time_period", "week")
            limit = arguments.get("limit", 20)
            
            movies = await movie_db.get_trending_movies(limit, time_period)
            
            result = f"📈 **Trending Movies This {time_period.title()}**\n\n"
            for i, movie in enumerate(movies, 1):
                result += f"{i}. **{movie['title']}** ({str(movie.get('release_date', 'Unknown'))[:4]})\n"
                result += f"   ⭐ {movie['vote_average']}/10 | 🔥 Trending Score: {movie.get('trending_score', 0):.1f}\n"
                result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "find_similar_movies":
            movie_id = arguments.get("movie_id")
            limit = arguments.get("limit", 10)
            
            # Get the source movie details
            source_movie = await movie_server.get_enhanced_movie_details(movie_id)
            
            # Find similar movies based on content
            similar_movies = await movie_db.get_content_based_recommendations("demo_user", limit)
            
            result = f"🎭 **Movies Similar to '{source_movie['title']}'**\n\n"
            result += f"Based on: {', '.join(source_movie.get('genres', []))}\n\n"
            
            for i, movie in enumerate(similar_movies, 1):
                result += f"{i}. **{movie['title']}** ({str(movie.get('release_date', 'Unknown'))[:4]})\n"
                result += f"   ⭐ Rating: {movie['vote_average']}/10\n"
                result += f"   🎯 Similarity Score: {movie.get('score', 0):.1f}\n"
                result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "mood_based_search":
            query = arguments.get("query")
            username = arguments.get("username", "demo_user")
            limit = arguments.get("limit", 10)
            
            # Get recommendations
            recommendations = await movie_server.get_mood_based_recommendations(query, username, limit)
            
            # Format response
            result = f"🎭 **Mood-Based Recommendations for: '{query}'**\n\n"
            
            if not recommendations:
                result += "No movies found matching your mood. Try different keywords!"
            else:
                # Show what we understood
                analysis = movie_server.mood_analyzer.analyze_query(query)
                if analysis['moods']:
                    result += f"**Detected Mood:** {', '.join(analysis['moods'])}\n"
                if analysis['genres']:
                    result += f"**Looking for:** {', '.join(set(analysis['genres']))} movies\n"
                if analysis.get('context'):
                    context_info = []
                    if 'time_preference' in analysis['context']:
                        context_info.append(f"Time: {analysis['context']['time_preference']}")
                    if 'viewer_count' in analysis['context']:
                        context_info.append(f"Viewers: {analysis['context']['viewer_count']}")
                    if context_info:
                        result += f"**Context:** {', '.join(context_info)}\n"
                result += "\n"
                
                # Show recommendations
                for i, movie in enumerate(recommendations, 1):
                    result += f"{i}. **{movie['title']}** ({str(movie.get('release_date', 'Unknown'))[:4]})\n"
                    result += f"   ⭐ Rating: {movie['vote_average']}/10"
                    
                    # Add mood score
                    if 'mood_score' in movie:
                        result += f" | 🎯 Mood Match: {movie['mood_score']:.1f}/5"
                    
                    # Add sentiment if available
                    if movie.get('sentiment_analysis') and movie['sentiment_analysis']['review_count'] > 0:
                        sentiment = movie['sentiment_analysis']['overall_sentiment']
                        emoji = {"very_positive": "😍", "positive": "😊", "mixed": "😐", 
                                "negative": "😕", "very_negative": "😞"}.get(sentiment, "🤔")
                        result += f" | {emoji} {sentiment.replace('_', ' ').title()}"
                    
                    # Add runtime if relevant to query
                    if movie.get('runtime') and any(word in query.lower() for word in ['short', 'long', 'quick']):
                        result += f" | ⏱️ {movie['runtime']} min"
                    
                    result += f"\n   🎬 {movie.get('overview', '')[:100]}...\n"
                    
                    # Show why it matched
                    if movie.get('mood_match') and movie['mood_match']['matched_genres']:
                        result += f"   💡 Matches: {', '.join(movie['mood_match']['matched_genres'])}\n"
                    
                    result += f"   🆔 ID: {movie['movie_id']}\n\n"
            
            return [types.TextContent(type="text", text=result)]
        
        elif name == "analyze_movie_sentiment":
            movie_id = arguments.get("movie_id")
            
            # Get movie details
            movie = await movie_server.get_enhanced_movie_details(movie_id)
            
            # Get sentiment analysis
            async with movie_db.pool.acquire() as conn:
                sentiment = await movie_server.sentiment_analyzer.analyze_movie_sentiment(movie_id, conn)
            
            # Format response
            result = f"😊 **Sentiment Analysis for '{movie['title']}'**\n\n"
            
            if sentiment['review_count'] == 0:
                result += "No reviews available for sentiment analysis.\n"
                result += "Users need to add reviews for this movie first!\n\n"
                result += "💡 **Tip:** Use the `rate_movie` tool with a review to add sentiment data."
            else:
                # Overall sentiment with emoji
                sentiment_emoji = {
                    "very_positive": "😍", 
                    "positive": "😊", 
                    "mixed": "😐", 
                    "negative": "😕", 
                    "very_negative": "😞"
                }.get(sentiment['overall_sentiment'], "🤔")
                
                result += f"**Overall Sentiment:** {sentiment_emoji} {sentiment['overall_sentiment'].replace('_', ' ').title()}\n"
                result += f"**Based on:** {sentiment['review_count']} reviews\n"
                result += f"**Average Polarity:** {sentiment['average_polarity']} "
                
                # Explain polarity
                if sentiment['average_polarity'] > 0.5:
                    result += "(Very Positive)\n\n"
                elif sentiment['average_polarity'] > 0:
                    result += "(Positive)\n\n"
                elif sentiment['average_polarity'] < -0.5:
                    result += "(Very Negative)\n\n"
                elif sentiment['average_polarity'] < 0:
                    result += "(Negative)\n\n"
                else:
                    result += "(Neutral)\n\n"
                
                # Sentiment distribution
                result += "**📊 Sentiment Distribution:**\n"
                dist = sentiment['sentiment_distribution']
                total = sum(dist.values())
                
                # Create visual bars
                for sent_type in ['positive', 'negative', 'neutral']:
                    count = dist.get(sent_type, 0)
                    percent = (count / total) * 100 if total > 0 else 0
                    bar_length = int(percent / 10)  # Scale to 10 chars
                    bar = "█" * bar_length + "░" * (10 - bar_length)
                    emoji = {"positive": "😊", "negative": "😕", "neutral": "😐"}.get(sent_type, "")
                    result += f"{emoji} {sent_type.title()}: {bar} {count} ({percent:.1f}%)\n"
                
                # Top discussed aspects
                if sentiment.get('top_aspects'):
                    result += "\n**🎬 Most Discussed Aspects:**\n"
                    aspect_emojis = {
                        "acting": "🎭",
                        "story": "📖",
                        "direction": "🎬",
                        "visuals": "🎨",
                        "music": "🎵",
                        "pacing": "⏱️",
                        "emotion": "❤️"
                    }
                    for aspect, count in sentiment['top_aspects']:
                        emoji = aspect_emojis.get(aspect, "•")
                        result += f"{emoji} {aspect.title()}: mentioned {count} times\n"
                
                # Sample reviews if available
                result += "\n💡 **Tip:** Add more reviews to improve sentiment accuracy!"
            
            return [types.TextContent(type="text", text=result)]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}"
            )]
    
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"❌ Error: {str(e)}\n\nPlease check the logs for more details."
        )]


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