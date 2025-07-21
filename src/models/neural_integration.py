#!/usr/bin/env python3
"""
Neural Recommendations MCP Integration
=====================================
Integrates the trained neural collaborative filtering model with the 
existing MCP server to provide real-time neural recommendations.

Features:
- Load trained neural model
- Real-time neural recommendations 
- User/movie embedding similarity search
- Performance comparison with existing methods
- Caching for fast responses
- Seamless integration with existing MCP tools
"""

import asyncio
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
import time
from datetime import datetime, timedelta
import hashlib

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our models
try:
    from src.models.neural_recommender import AdvancedNeuralRecommender
    from src.models.database import movie_db, initialize_database
except ImportError:
    sys.path.append('.')
    from models.neural_recommender import AdvancedNeuralRecommender
    from models.database import movie_db, initialize_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralRecommendationService:
    """
    Production-ready neural recommendation service for MCP integration.
    
    This service loads the trained neural model and provides fast,
    cached recommendations for the MCP server.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the neural recommendation service.
        
        Args:
            model_path: Path to the trained neural model (auto-detected if None)
        """
        # Auto-detect model path
        if model_path is None:
            # Check if we're in src/ directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if 'src' in current_dir:
                # Go up to project root
                project_root = os.path.dirname(os.path.dirname(current_dir))
                self.model_path = os.path.join(project_root, "neural_models", "advanced_neural_recommender")
            else:
                self.model_path = "neural_models/advanced_neural_recommender"
        else:
            self.model_path = model_path
            
        self.model = None
        self.is_loaded = False
        self._load_attempted = False
        
        # Performance tracking
        self.recommendation_cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0,
            'neural_recommendations_served': 0
        }
        
        logger.info(f"🧠 Neural service initialized with model path: {self.model_path}")
        
        # NOTE: Don't load model here - do it lazily when first needed
    
    async def ensure_model_loaded(self):
        """Ensure the model is loaded before use."""
        if not self.is_loaded and not self._load_attempted:
            self._load_attempted = True
            logger.info(f"🔄 Attempting to load neural model for first time...")
            await self.load_model()
        elif not self.is_loaded:
            logger.warning(f"⚠️ Model loading was attempted before but failed")
        return self.is_loaded
    
    async def load_model(self):
        """Load the trained neural collaborative filtering model."""
        try:
            logger.info(f"🧠 Loading neural model from {self.model_path}...")
            
            # Enable unsafe deserialization for Lambda layers
            import tensorflow as tf
            tf.keras.config.enable_unsafe_deserialization()
            
            # Check if model files exist (try .keras first, then .h5)
            keras_model_file = f"{self.model_path}_model.keras"
            h5_model_file = f"{self.model_path}_model.h5"
            metadata_file = f"{self.model_path}_metadata.pkl"
            
            # Check if files exist
            if not os.path.exists(metadata_file):
                logger.warning(f"Neural model metadata not found at {self.model_path}. Run training first!")
                logger.info(f"Expected file: {metadata_file}")
                return False
            
            if os.path.exists(keras_model_file):
                # New .keras format
                self.model = AdvancedNeuralRecommender.load_model(self.model_path)
            elif os.path.exists(h5_model_file):
                # Legacy .h5 format
                self.model = AdvancedNeuralRecommender.load_model(self.model_path)
                logger.warning("Loaded legacy .h5 model. Consider retraining for .keras format.")
            else:
                logger.warning(f"Neural model files not found at {self.model_path}. Run training first!")
                logger.info(f"Expected files: {keras_model_file} or {h5_model_file}")
                return False
            
            self.is_loaded = True
            
            logger.info("✅ Neural model loaded successfully!")
            logger.info(f"📊 Model Info:")
            logger.info(f"   Users: {self.model.num_users}")
            logger.info(f"   Movies: {self.model.num_movies}")
            logger.info(f"   Embedding Dimension: {self.model.embedding_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")
            self.is_loaded = False
            return False
    
    def _generate_cache_key(self, user_id: str, context: Dict, limit: int) -> str:
        """Generate a cache key for recommendations."""
        context_str = json.dumps(context, sort_keys=True)
        key_data = f"{user_id}_{context_str}_{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    async def get_user_features(self, user_id: str) -> Dict:
        """Get user features for neural model prediction."""
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            # Get user statistics
            user_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    VARIANCE(rating) as rating_variance,
                    COUNT(DISTINCT DATE(created_at)) as days_active
                FROM user_ratings
                WHERE user_id = $1
            """, user_id)
            
            if not user_stats or user_stats['total_ratings'] == 0:
                # Default features for new user
                return {
                    'user_activity_count': 1,
                    'user_avg_rating': 3.0,
                    'rating_variance': 1.0,
                    'user_rating_frequency': 1.0
                }
            
            days_active = max(user_stats['days_active'], 1)
            return {
                'user_activity_count': user_stats['total_ratings'],
                'user_avg_rating': float(user_stats['avg_rating']),
                'rating_variance': float(user_stats['rating_variance'] or 1.0),
                'user_rating_frequency': user_stats['total_ratings'] / days_active
            }
    
    async def resolve_user_id(self, username: str) -> str:
        """Convert username to actual user_id used in neural model."""
        logger.info(f"🔍 Resolving username: {username}")
        
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            # First, try to get user by username
            try:
                user = await conn.fetchrow("""
                    SELECT user_id, username FROM users WHERE username = $1
                """, username)
                
                if user:
                    # Convert UUID object to string properly
                    if hasattr(user['user_id'], 'hex'):
                        # It's a UUID object
                        user_id = str(user['user_id'])
                    else:
                        # It's already a string
                        user_id = str(user['user_id'])
                    
                    logger.info(f"✅ Found user in database: {username} -> {user_id}")
                    return user_id
                    
            except Exception as e:
                logger.error(f"❌ Database query failed: {e}")
            
            # If user doesn't exist, check if the input is already a UUID
            try:
                import uuid
                # Try to parse as UUID - if successful, it's already a UUID
                parsed_uuid = uuid.UUID(username)
                uuid_str = str(parsed_uuid)
                logger.info(f"✅ Input {username} is already a valid UUID: {uuid_str}")
                return uuid_str
            except (ValueError, TypeError):
                pass
            
            # If we reach here, user doesn't exist - check if they're in the model anyway
            if self.model and str(username) in self.model.user_encoder:
                logger.info(f"✅ Username {username} found directly in model encoders")
                return str(username)
                
            # Last resort: create user if doesn't exist
            logger.warning(f"⚠️ User {username} not found anywhere, using collaborative filtering fallback")
            return str(username)  # Return as string for fallback handling
    
    async def get_movie_features(self, movie_ids: List[int]) -> pd.DataFrame:
        """Get movie features for neural model prediction."""
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            # Get movie details
            movies_data = await conn.fetch("""
                SELECT 
                    m.movie_id,
                    m.title,
                    m.genres,
                    m.release_date,
                    m.vote_average,
                    m.popularity,
                    m.runtime,
                    COALESCE(user_stats.total_user_ratings, 0) as total_user_ratings,
                    COALESCE(user_stats.avg_user_rating, m.vote_average) as avg_user_rating
                FROM movies m
                LEFT JOIN (
                    SELECT 
                        movie_id,
                        COUNT(*) as total_user_ratings,
                        AVG(rating) as avg_user_rating
                    FROM user_ratings
                    GROUP BY movie_id
                ) user_stats ON m.movie_id = user_stats.movie_id
                WHERE m.movie_id = ANY($1::int[])
            """, movie_ids)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in movies_data])
            
            if df.empty:
                return pd.DataFrame()
            
            # Process features
            df['release_year'] = df['release_date'].str[:4]
            df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2000)
            df['popularity'] = df['popularity'].fillna(0)
            df['vote_average'] = df['vote_average'].fillna(df['vote_average'].median())
            df['runtime'] = df['runtime'].fillna(df['runtime'].median())
            
            return df
    
    async def get_neural_recommendations(
        self, 
        user_id: str, 
        limit: int = 10,
        exclude_rated: bool = True,
        context: Dict = None
    ) -> List[Dict]:
        """
        Get neural collaborative filtering recommendations.
        
        Args:
            user_id: User ID or username for recommendations
            limit: Number of recommendations to return
            exclude_rated: Whether to exclude already rated movies
            context: Additional context for recommendations
            
        Returns:
            List of recommended movies with prediction scores
        """
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        logger.info(f"🎯 Neural recommendation request for user: {user_id}")
        
        # Ensure model is loaded
        logger.info(f"📋 Checking if model is loaded...")
        if not await self.ensure_model_loaded():
            logger.warning(f"❌ Neural model not loaded. Falling back to collaborative filtering for {user_id}")
            return await self._fallback_recommendations(user_id, limit)

        logger.info(f"✅ Neural model is loaded successfully!")

        try:
            # Resolve username to actual user_id FIRST, before any database calls
            logger.info(f"🔍 Resolving user ID for: {user_id}")
            actual_user_id = await self.resolve_user_id(user_id)
            logger.info(f"🆔 Resolved to actual user_id: {actual_user_id}")
            
            # Check cache using resolved user_id
            context = context or {}
            cache_key = self._generate_cache_key(actual_user_id, context, limit)
            
            if cache_key in self.recommendation_cache:
                cache_entry = self.recommendation_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.performance_stats['cache_hits'] += 1
                    logger.info(f"🚀 Cache hit for user {user_id}")
                    return cache_entry['recommendations']

            # Check if user exists in neural model
            if actual_user_id not in self.model.user_encoder:
                logger.warning(f"❌ User {user_id} (ID: {actual_user_id}) not found in neural model training data")
                logger.info(f"📊 Model has {len(self.model.user_encoder)} users")
                
                # Show sample of actual user IDs in model for debugging
                sample_user_ids = list(self.model.user_encoder.keys())[:3]
                logger.info(f"📋 Sample user IDs in model: {sample_user_ids}")
                
                # Check if any UUID version exists
                import uuid
                try:
                    uuid_version = str(uuid.UUID(actual_user_id))
                    if uuid_version in self.model.user_encoder:
                        logger.info(f"🔄 Found UUID version: {uuid_version}")
                        actual_user_id = uuid_version
                    else:
                        logger.warning(f"❌ UUID version {uuid_version} also not in model")
                        return await self._fallback_recommendations(user_id, limit)
                except:
                    logger.warning(f"❌ Could not convert {actual_user_id} to UUID format")
                    return await self._fallback_recommendations(user_id, limit)
            
            logger.info(f"✅ User found in neural model!")
            
            # Get user features using resolved UUID
            user_features = await self.get_user_features(actual_user_id)
            logger.info(f"📊 User features: {user_features}")
            
            # Get candidate movies using resolved UUID
            candidate_movies = await self._get_candidate_movies(actual_user_id, exclude_rated, limit * 3)
            logger.info(f"🎬 Found {len(candidate_movies)} candidate movies")
            
            if not candidate_movies:
                logger.warning(f"❌ No candidate movies found for user {user_id}")
                return await self._fallback_recommendations(user_id, limit)
            
            # Get movie features
            movie_ids = [m['movie_id'] for m in candidate_movies]
            movie_features_df = await self.get_movie_features(movie_ids)
            logger.info(f"🎞️ Movie features shape: {movie_features_df.shape if not movie_features_df.empty else 'empty'}")
            
            if movie_features_df.empty:
                logger.warning(f"❌ No movie features found")
                return await self._fallback_recommendations(user_id, limit)
            
            # Prepare features for neural model (use actual_user_id)
            features_df = self._prepare_prediction_features(actual_user_id, user_features, movie_features_df)
            logger.info(f"🔧 Prepared features shape: {features_df.shape}")
            
            # Get neural predictions
            logger.info(f"🧠 Making neural predictions...")
            predictions = self.model.predict(
                user_ids=[actual_user_id] * len(features_df),
                movie_ids=movie_ids,
                features_df=features_df
            )
            logger.info(f"📈 Generated {len(predictions)} predictions")
            
            # Combine predictions with movie data
            recommendations = []
            for i, (movie, pred_rating) in enumerate(zip(candidate_movies, predictions)):
                movie_data = movie.copy()
                movie_data.update({
                    'predicted_rating': float(pred_rating),
                    'confidence_score': self._calculate_confidence(user_features, movie),
                    'recommendation_source': 'neural_collaborative_filtering',
                    'user_id': user_id  # Return original username
                })
                recommendations.append(movie_data)
            
            # Sort by predicted rating and limit results
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            final_recommendations = recommendations[:limit]
            
            # Cache results using actual_user_id for consistency
            self.recommendation_cache[cache_key] = {
                'recommendations': final_recommendations,
                'timestamp': time.time()
            }
            
            # Update performance stats
            response_time = time.time() - start_time
            self.performance_stats['average_response_time'] = (
                self.performance_stats['average_response_time'] + response_time
            ) / 2
            self.performance_stats['neural_recommendations_served'] += len(final_recommendations)
            
            logger.info(f"🎉 Neural recommendations for {user_id}: {len(final_recommendations)} movies in {response_time:.3f}s")
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"💥 Neural recommendation failed for user {user_id}: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return await self._fallback_recommendations(user_id, limit)
    
    def _prepare_prediction_features(self, user_id: str, user_features: Dict, movie_features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features DataFrame for neural model prediction."""
        
        # Create features for each movie
        features_list = []
        
        for _, movie in movie_features_df.iterrows():
            features = {
                'user_id': user_id,
                'movie_id': movie['movie_id'],
                'rating': 0,  # Placeholder for prediction
                'genres': movie['genres'],
                'release_year': movie['release_year'],
                'popularity': movie['popularity'],
                'vote_average': movie['vote_average'],
                'user_activity_count': user_features['user_activity_count'],
                'user_avg_rating': user_features['user_avg_rating'],
                'rating_variance': user_features['rating_variance'],
                'runtime': movie['runtime'],
                'movie_age_when_rated': 2025 - movie['release_year'],  # Current year
                'user_rating_frequency': user_features['user_rating_frequency'],
                'is_favorite': 0,
                'is_high_rating': 0,
                'is_low_rating': 0
            }
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_confidence(self, user_features: Dict, movie: Dict) -> float:
        """Calculate confidence score for a recommendation."""
        
        # Base confidence on user activity
        activity_confidence = min(user_features['user_activity_count'] / 20.0, 1.0)
        
        # Movie popularity confidence
        popularity_confidence = min(movie.get('total_user_ratings', 0) / 50.0, 1.0)
        
        # Combined confidence (average)
        confidence = (activity_confidence + popularity_confidence) / 2
        
        return float(confidence)
    
    async def _get_candidate_movies(self, user_id: str, exclude_rated: bool, limit: int) -> List[Dict]:
        """Get candidate movies for recommendation."""
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            if exclude_rated:
                # Exclude movies user has already rated
                movies = await conn.fetch("""
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.genres,
                        m.release_date,
                        m.vote_average,
                        m.popularity,
                        m.runtime,
                        COALESCE(user_stats.total_user_ratings, 0) as total_user_ratings
                    FROM movies m
                    LEFT JOIN (
                        SELECT movie_id, COUNT(*) as total_user_ratings
                        FROM user_ratings
                        GROUP BY movie_id
                    ) user_stats ON m.movie_id = user_stats.movie_id
                    WHERE m.vote_average >= 6.0
                        AND m.movie_id NOT IN (
                            SELECT movie_id FROM user_ratings WHERE user_id = $1
                        )
                    ORDER BY (m.vote_average * 0.7 + COALESCE(m.popularity, 0) * 0.3 / 1000) DESC
                    LIMIT $2
                """, user_id, limit)
            else:
                # Include all movies
                movies = await conn.fetch("""
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.genres,
                        m.release_date,
                        m.vote_average,
                        m.popularity,
                        m.runtime,
                        COALESCE(user_stats.total_user_ratings, 0) as total_user_ratings
                    FROM movies m
                    LEFT JOIN (
                        SELECT movie_id, COUNT(*) as total_user_ratings
                        FROM user_ratings
                        GROUP BY movie_id
                    ) user_stats ON m.movie_id = user_stats.movie_id
                    WHERE m.vote_average >= 6.0
                    ORDER BY (m.vote_average * 0.7 + COALESCE(m.popularity, 0) * 0.3 / 1000) DESC
                    LIMIT $1
                """, limit)
            
            return [dict(movie) for movie in movies]
    
    async def _fallback_recommendations(self, user_id: str, limit: int) -> List[Dict]:
        """Fallback to collaborative filtering if neural model fails."""
        logger.info(f"🔄 Using collaborative filtering fallback for {user_id}")
        
        try:
            # For fallback, try to get popular movies - safer than user-specific queries
            await initialize_database()
            async with movie_db.pool.acquire() as conn:
                popular_movies = await conn.fetch("""
                    SELECT 
                        movie_id,
                        title,
                        vote_average,
                        release_date,
                        genres,
                        overview,
                        popularity
                    FROM movies 
                    WHERE vote_average >= 7.0 
                    ORDER BY popularity DESC 
                    LIMIT $1
                """, limit)
                
                result = []
                for movie in popular_movies:
                    movie_data = dict(movie)
                    movie_data.update({
                        'predicted_rating': float(movie['vote_average']) / 2,  # Convert 10-scale to 5-scale
                        'confidence_score': 0.3,
                        'recommendation_source': 'popular_fallback'
                    })
                    result.append(movie_data)
                
                logger.info(f"✅ Fallback: returning {len(result)} popular movies")
                return result
            
        except Exception as e:
            logger.error(f"❌ Even fallback failed: {e}")
            return []
    
    async def find_similar_users_neural(self, user_id: str, top_k: int = 10) -> List[Dict]:
        """Find similar users using neural embeddings."""
        if not await self.ensure_model_loaded():
            return []
        
        try:
            # Resolve username to actual user_id
            actual_user_id = await self.resolve_user_id(user_id)
            
            # Check if user exists in neural model
            if actual_user_id not in self.model.user_encoder:
                logger.info(f"User {user_id} not found in neural model training data")
                return []
            
            similar_users = self.model.find_similar_users(actual_user_id, top_k)
            
            # Get user details and convert back to usernames
            result = []
            for similar_user_id, similarity in similar_users:
                # Try to find username for this user_id
                await initialize_database()
                async with movie_db.pool.acquire() as conn:
                    user_info = await conn.fetchrow("""
                        SELECT username FROM users WHERE user_id = $1
                    """, similar_user_id)
                
                display_user_id = user_info['username'] if user_info else similar_user_id
                user_features = await self.get_user_features(similar_user_id)
                
                result.append({
                    'user_id': display_user_id,
                    'similarity_score': similarity,
                    'total_ratings': user_features['user_activity_count'],
                    'avg_rating': user_features['user_avg_rating'],
                    'method': 'neural_embeddings'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Neural user similarity search failed: {e}")
            return []
    
    async def find_similar_movies_neural(self, movie_id: int, top_k: int = 10) -> List[Dict]:
        """Find similar movies using neural embeddings."""
        if not await self.ensure_model_loaded():
            return []
        
        try:
            similar_movies = self.model.find_similar_movies(movie_id, top_k)
            
            # Get movie details
            movie_ids = [mid for mid, _ in similar_movies]
            movie_features_df = await self.get_movie_features(movie_ids)
            
            result = []
            for movie_id, similarity in similar_movies:
                movie_data = movie_features_df[movie_features_df['movie_id'] == movie_id]
                if not movie_data.empty:
                    movie_info = movie_data.iloc[0].to_dict()
                    movie_info.update({
                        'similarity_score': similarity,
                        'method': 'neural_embeddings'
                    })
                    result.append(movie_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Neural movie similarity search failed: {e}")
            return []
    
    async def compare_recommendation_methods(self, user_id: str, limit: int = 10) -> Dict:
        """Compare neural vs collaborative filtering recommendations."""
        
        # Get neural recommendations
        neural_recs = await self.get_neural_recommendations(user_id, limit)
        
        # Get collaborative filtering recommendations
        collab_recs = await movie_db.get_collaborative_recommendations(user_id, limit)
        
        # Add source tags
        for rec in neural_recs:
            rec['method'] = 'neural_embeddings'
        
        for rec in collab_recs:
            rec['method'] = 'collaborative_filtering'
        
        # Find overlap
        neural_movie_ids = {rec['movie_id'] for rec in neural_recs}
        collab_movie_ids = {rec['movie_id'] for rec in collab_recs}
        
        overlap = neural_movie_ids & collab_movie_ids
        overlap_percentage = len(overlap) / max(len(neural_movie_ids), 1) * 100
        
        return {
            'neural_recommendations': neural_recs,
            'collaborative_recommendations': collab_recs,
            'overlap_count': len(overlap),
            'overlap_percentage': overlap_percentage,
            'unique_to_neural': len(neural_movie_ids - collab_movie_ids),
            'unique_to_collaborative': len(collab_movie_ids - neural_movie_ids),
            'comparison_timestamp': datetime.now().isoformat()
        }
    
    def get_performance_stats(self) -> Dict:
        """Get service performance statistics."""
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / max(self.performance_stats['total_requests'], 1)
        ) * 100
        
        stats = {
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'load_attempted': self._load_attempted,
            'total_requests': self.performance_stats['total_requests'],
            'cache_hit_rate': cache_hit_rate,
            'average_response_time': self.performance_stats['average_response_time'],
            'neural_recommendations_served': self.performance_stats['neural_recommendations_served'],
            'cache_size': len(self.recommendation_cache),
            'model_info': {
                'num_users': self.model.num_users if self.model else 0,
                'num_movies': self.model.num_movies if self.model else 0,
                'embedding_dim': self.model.embedding_dim if self.model else 0
            } if self.is_loaded else {}
        }
        
        # Add debug info if model is loaded
        if self.is_loaded and self.model:
            # Sample of user IDs in the model (first 5)
            sample_user_ids = list(self.model.user_encoder.keys())[:5]
            sample_movie_ids = list(self.model.movie_encoder.keys())[:5]
            
            stats['debug_info'] = {
                'sample_user_ids': sample_user_ids,
                'sample_movie_ids': sample_movie_ids,
                'total_encoded_users': len(self.model.user_encoder),
                'total_encoded_movies': len(self.model.movie_encoder)
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the recommendation cache."""
        self.recommendation_cache.clear()
        logger.info("🧹 Recommendation cache cleared")


# Global service instance - HARDCODE PATH FOR TESTING
neural_service = NeuralRecommendationService(
    model_path=r"C:\Users\bhati\movie-discovery-mcp\neural_models\advanced_neural_recommender"
)


# ============= MCP TOOL INTEGRATION FUNCTIONS =============

async def get_neural_recommendations_tool(username: str, limit: int = 10, compare_methods: bool = False) -> List[Dict]:
    """
    MCP tool for getting neural collaborative filtering recommendations.
    
    Args:
        username: Username for recommendations
        limit: Number of recommendations
        compare_methods: Whether to compare with collaborative filtering
        
    Returns:
        List of recommendations or comparison results
    """
    if compare_methods:
        comparison = await neural_service.compare_recommendation_methods(username, limit)
        return comparison
    else:
        recommendations = await neural_service.get_neural_recommendations(username, limit)
        return recommendations


async def find_similar_users_neural_tool(user_id: str, top_k: int = 10) -> List[Dict]:
    """
    MCP tool for finding similar users using neural embeddings.
    """
    return await neural_service.find_similar_users_neural(user_id, top_k)


async def find_similar_movies_neural_tool(movie_id: int, top_k: int = 10) -> List[Dict]:
    """
    MCP tool for finding similar movies using neural embeddings.
    """
    return await neural_service.find_similar_movies_neural(movie_id, top_k)


async def get_neural_service_stats_tool() -> Dict:
    """
    MCP tool for getting neural service performance statistics.
    """
    return neural_service.get_performance_stats()


async def clear_neural_cache_tool() -> Dict:
    """
    MCP tool for clearing the neural recommendation cache.
    """
    neural_service.clear_cache()
    return {"status": "success", "message": "Neural recommendation cache cleared"}


if __name__ == "__main__":
    # Test the neural service
    async def test_service():
        print("🧠 Testing Neural Recommendation Service")
        print("=" * 40)
        
        # Load model
        success = await neural_service.load_model()
        if success:
            print("✅ Model loaded successfully!")
            
            # Test recommendations
            recs = await neural_service.get_neural_recommendations("demo_user", limit=5)
            print(f"🎬 Got {len(recs)} recommendations")
            
            # Test performance stats
            stats = neural_service.get_performance_stats()
            print(f"📊 Performance: {stats}")
        else:
            print("❌ Model loading failed. Train the model first!")
    
    asyncio.run(test_service())