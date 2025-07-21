#!/usr/bin/env python3
"""
Advanced Neural Collaborative Filtering for Movie Recommendations
================================================================
A sophisticated neural network that learns user and movie embeddings
with additional features for enhanced recommendation accuracy.

Architecture:
- User Embeddings: 128-dimensional learned representations
- Movie Embeddings: 128-dimensional learned representations  
- Side Features: Genres, release year, vote average, user patterns
- Multi-layer Neural Network: 3 hidden layers with dropout
- Output: Predicted rating (1-5 scale)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class AdvancedNeuralRecommender:
    """
    Advanced neural collaborative filtering model with side features.
    
    This is NOT your basic matrix factorization. This is production-grade
    neural recommendation system with learned embeddings and feature engineering.
    """
    
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-5,
        learning_rate: float = 0.001
    ):
        """
        Initialize the neural recommender.
        
        Args:
            num_users: Total number of unique users
            num_movies: Total number of unique movies
            embedding_dim: Dimension of user/movie embeddings
            hidden_dims: Hidden layer dimensions [256, 128, 64]
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Adam optimizer learning rate
        """
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        # Model components
        self.model = None
        self.user_encoder = {}  # user_id -> internal_id mapping
        self.movie_encoder = {}  # movie_id -> internal_id mapping
        self.user_decoder = {}  # internal_id -> user_id mapping
        self.movie_decoder = {}  # internal_id -> movie_id mapping
        
        # Feature processing
        self.genre_encoder = {}
        self.year_normalizer = None
        self.rating_normalizer = None
        
        # Model metadata
        self.is_trained = False
        self.training_history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the advanced neural collaborative filtering model.
        
        Architecture:
        1. Embedding layers for users and movies
        2. Additional feature inputs (genres, year, etc.)
        3. Multi-layer neural network with dropout
        4. Regularization to prevent overfitting
        5. Output layer for rating prediction
        """
        
        # ============= INPUT LAYERS =============
        # Primary inputs
        user_input = layers.Input(shape=(), name='user_id', dtype='int32')
        movie_input = layers.Input(shape=(), name='movie_id', dtype='int32')
        
        # Additional feature inputs
        genre_input = layers.Input(shape=(20,), name='genre_features', dtype='float32')  # One-hot genres
        year_input = layers.Input(shape=(), name='release_year', dtype='float32')
        popularity_input = layers.Input(shape=(), name='movie_popularity', dtype='float32')
        rating_avg_input = layers.Input(shape=(), name='movie_rating_avg', dtype='float32')
        user_activity_input = layers.Input(shape=(), name='user_activity', dtype='float32')
        
        # ============= EMBEDDING LAYERS =============
        # User embeddings with L2 regularization
        user_embedding = layers.Embedding(
            self.num_users, 
            self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='user_embedding'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        # Movie embeddings with L2 regularization
        movie_embedding = layers.Embedding(
            self.num_movies, 
            self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='movie_embedding'
        )(movie_input)
        movie_embedding = layers.Flatten()(movie_embedding)
        
        # ============= BIAS TERMS =============
        # User and movie bias (like in matrix factorization)
        user_bias = layers.Embedding(self.num_users, 1, name='user_bias')(user_input)
        user_bias = layers.Flatten()(user_bias)
        
        movie_bias = layers.Embedding(self.num_movies, 1, name='movie_bias')(movie_input)
        movie_bias = layers.Flatten()(movie_bias)
        
        # ============= FEATURE PROCESSING =============
        # Process additional features
        year_processed = layers.Dense(8, activation='relu', name='year_processing')(
            layers.Reshape((1,))(year_input)
        )
        
        popularity_processed = layers.Dense(8, activation='relu', name='popularity_processing')(
            layers.Reshape((1,))(popularity_input)
        )
        
        rating_processed = layers.Dense(8, activation='relu', name='rating_processing')(
            layers.Reshape((1,))(rating_avg_input)
        )
        
        activity_processed = layers.Dense(8, activation='relu', name='activity_processing')(
            layers.Reshape((1,))(user_activity_input)
        )
        
        # Process genre features
        genre_processed = layers.Dense(16, activation='relu', name='genre_processing')(genre_input)
        
        # ============= COMBINE FEATURES =============
        # Concatenate all features
        combined_features = layers.Concatenate(name='feature_concatenation')([
            user_embedding,
            movie_embedding,
            genre_processed,
            year_processed,
            popularity_processed,
            rating_processed,
            activity_processed
        ])
        
        # ============= DEEP NEURAL NETWORK =============
        # Multi-layer neural network with dropout
        x = combined_features
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = layers.Dense(
                hidden_dim, 
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'hidden_layer_{i+1}'
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            
            # Batch normalization for better training
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
        
        # ============= OUTPUT LAYER =============
        # Combine neural network output with bias terms
        neural_output = layers.Dense(1, name='neural_output')(x)
        
        # Final prediction: neural network + biases
        output = layers.Add(name='final_prediction')([
            neural_output,
            user_bias,
            movie_bias
        ])
        
        # Ensure output is in valid rating range (1-5)
        output = layers.Activation('sigmoid', name='sigmoid_activation')(output)
        output = layers.Lambda(lambda x: x * 4 + 1, name='rating_scaling')(output)  # Scale to 1-5
        
        # ============= CREATE MODEL =============
        model = keras.Model(
            inputs=[
                user_input, movie_input, genre_input, year_input,
                popularity_input, rating_avg_input, user_activity_input
            ],
            outputs=output,
            name='AdvancedNeuralRecommender'
        )
        
        # ============= COMPILE MODEL =============
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def prepare_encoders(self, user_ids: List, movie_ids: List):
        """
        Create user and movie ID encoders for the neural network.
        Neural networks need consecutive integer IDs starting from 0.
        """
        # Create user encoders
        unique_users = sorted(list(set(user_ids)))
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        
        # Create movie encoders  
        unique_movies = sorted(list(set(movie_ids)))
        self.movie_encoder = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.movie_decoder = {idx: movie_id for movie_id, idx in self.movie_encoder.items()}
        
        # Update model parameters
        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)
        
        logger.info(f"Encoders prepared: {self.num_users} users, {self.num_movies} movies")
    
    def encode_features(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Encode features for neural network input.
        
        Args:
            features_df: DataFrame with columns:
                - user_id, movie_id, rating (target)
                - genres, release_year, popularity, vote_average
                - user_activity_count
        
        Returns:
            Dictionary of encoded features ready for training
        """
        # Encode user and movie IDs
        encoded_users = [self.user_encoder[uid] for uid in features_df['user_id']]
        encoded_movies = [self.movie_encoder[mid] for mid in features_df['movie_id']]
        
        # Process genre features (one-hot encoding)
        genre_features = self._encode_genres(features_df['genres'])
        
        # Normalize numerical features
        years = self._normalize_years(features_df['release_year'])
        popularity = self._normalize_feature(features_df['popularity'])
        rating_avg = self._normalize_feature(features_df['vote_average'])
        user_activity = self._normalize_feature(features_df['user_activity_count'])
        
        return {
            'user_id': np.array(encoded_users, dtype=np.int32),
            'movie_id': np.array(encoded_movies, dtype=np.int32),
            'genre_features': genre_features,
            'release_year': years,
            'movie_popularity': popularity,
            'movie_rating_avg': rating_avg,
            'user_activity': user_activity,
            'ratings': np.array(features_df['rating'], dtype=np.float32)
        }
    
    def _encode_genres(self, genres_series: pd.Series) -> np.ndarray:
        """
        One-hot encode movie genres.
        
        Args:
            genres_series: Series of genre lists (from JSON)
            
        Returns:
            One-hot encoded genre matrix
        """
        # Get all unique genres
        all_genres = set()
        for genres_json in genres_series:
            if pd.notna(genres_json) and genres_json != 'null':
                try:
                    if isinstance(genres_json, str):
                        import json
                        genres = json.loads(genres_json)
                    else:
                        genres = genres_json
                    
                    if isinstance(genres, list):
                        for genre in genres:
                            if isinstance(genre, dict) and 'name' in genre:
                                all_genres.add(genre['name'])
                            elif isinstance(genre, str):
                                all_genres.add(genre)
                except:
                    continue
        
        # Create genre encoder if not exists
        if not self.genre_encoder:
            self.genre_encoder = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
        
        # One-hot encode
        num_genres = len(self.genre_encoder)
        genre_matrix = np.zeros((len(genres_series), max(20, num_genres)))
        
        for i, genres_json in enumerate(genres_series):
            if pd.notna(genres_json) and genres_json != 'null':
                try:
                    if isinstance(genres_json, str):
                        import json
                        genres = json.loads(genres_json)
                    else:
                        genres = genres_json
                    
                    if isinstance(genres, list):
                        for genre in genres:
                            genre_name = genre['name'] if isinstance(genre, dict) else genre
                            if genre_name in self.genre_encoder:
                                genre_matrix[i, self.genre_encoder[genre_name]] = 1.0
                except:
                    continue
        
        return genre_matrix[:, :20]  # Limit to top 20 genres
    
    def _normalize_years(self, years_series: pd.Series) -> np.ndarray:
        """Normalize release years to 0-1 range."""
        
        # Handle different data types for years
        def extract_year(date_val):
            if pd.isna(date_val):
                return None
            elif isinstance(date_val, str):
                return date_val[:4]
            elif hasattr(date_val, 'year'):  # datetime object
                return str(date_val.year)
            elif isinstance(date_val, (int, float)):
                return str(int(date_val))
            else:
                return str(date_val)[:4]
        
        # Extract years safely
        years = years_series.apply(extract_year)
        years = pd.to_numeric(years, errors='coerce')
        years = years.fillna(years.median())
        
        min_year, max_year = years.min(), years.max()
        
        # Avoid division by zero
        if max_year > min_year:
            normalized = (years - min_year) / (max_year - min_year)
        else:
            normalized = years * 0  # All zeros if no variance
        
        if self.year_normalizer is None:
            self.year_normalizer = (min_year, max_year)
        
        return normalized.values.astype(np.float32)
    
    def _normalize_feature(self, feature_series: pd.Series) -> np.ndarray:
        """Normalize numerical features using min-max scaling."""
        feature = pd.to_numeric(feature_series, errors='coerce').fillna(0)
        
        min_val, max_val = feature.min(), feature.max()
        if max_val > min_val:
            normalized = (feature - min_val) / (max_val - min_val)
        else:
            normalized = feature * 0  # All zeros if no variance
        
        return normalized.values.astype(np.float32)
    
    def train(
        self, 
        features_df: pd.DataFrame,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 512,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the neural collaborative filtering model.
        
        Args:
            features_df: Training data with all features
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info("Starting advanced neural collaborative filtering training...")
        
        # Prepare encoders
        self.prepare_encoders(features_df['user_id'].tolist(), features_df['movie_id'].tolist())
        
        # Encode features
        encoded_data = self.encode_features(features_df)
        
        # Build model
        self.model = self.build_model()
        
        logger.info(f"Model architecture:")
        self.model.summary()
        
        # Prepare training inputs
        X = {
            'user_id': encoded_data['user_id'],
            'movie_id': encoded_data['movie_id'],
            'genre_features': encoded_data['genre_features'],
            'release_year': encoded_data['release_year'],
            'movie_popularity': encoded_data['movie_popularity'],
            'movie_rating_avg': encoded_data['movie_rating_avg'],
            'user_activity': encoded_data['user_activity']
        }
        y = encoded_data['ratings']
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_neural_recommender.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history = history
        self.is_trained = True
        
        logger.info("Training completed!")
        return history
    
    def predict(self, user_ids: List, movie_ids: List, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for user-movie pairs.
        
        Args:
            user_ids: List of user IDs
            movie_ids: List of movie IDs  
            features_df: DataFrame with movie features
            
        Returns:
            Predicted ratings array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Encode features for prediction
        encoded_data = self.encode_features(features_df)
        
        # Prepare input data
        X = {
            'user_id': encoded_data['user_id'],
            'movie_id': encoded_data['movie_id'],
            'genre_features': encoded_data['genre_features'],
            'release_year': encoded_data['release_year'],
            'movie_popularity': encoded_data['movie_popularity'],
            'movie_rating_avg': encoded_data['movie_rating_avg'],
            'user_activity': encoded_data['user_activity']
        }
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        return predictions.flatten()
    
    def get_user_embedding(self, user_id) -> np.ndarray:
        """Get the learned embedding vector for a user."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if user_id not in self.user_encoder:
            raise ValueError(f"User {user_id} not found in training data")
        
        encoded_user = self.user_encoder[user_id]
        embedding_layer = self.model.get_layer('user_embedding')
        embedding = embedding_layer.get_weights()[0][encoded_user]
        
        return embedding
    
    def get_movie_embedding(self, movie_id) -> np.ndarray:
        """Get the learned embedding vector for a movie."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if movie_id not in self.movie_encoder:
            raise ValueError(f"Movie {movie_id} not found in training data")
        
        encoded_movie = self.movie_encoder[movie_id]
        embedding_layer = self.model.get_layer('movie_embedding')
        embedding = embedding_layer.get_weights()[0][encoded_movie]
        
        return embedding
    
    def find_similar_users(self, user_id, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find users with similar embeddings using cosine similarity.
        
        Args:
            user_id: Target user ID
            top_k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        target_embedding = self.get_user_embedding(user_id)
        
        # Get all user embeddings
        embedding_layer = self.model.get_layer('user_embedding')
        all_embeddings = embedding_layer.get_weights()[0]
        
        # Calculate cosine similarities
        similarities = []
        for encoded_user_id, embedding in enumerate(all_embeddings):
            original_user_id = self.user_decoder[encoded_user_id]
            if original_user_id != user_id:
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((original_user_id, float(similarity)))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_movies(self, movie_id, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find movies with similar embeddings."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        target_embedding = self.get_movie_embedding(movie_id)
        
        # Get all movie embeddings
        embedding_layer = self.model.get_layer('movie_embedding')
        all_embeddings = embedding_layer.get_weights()[0]
        
        # Calculate cosine similarities
        similarities = []
        for encoded_movie_id, embedding in enumerate(all_embeddings):
            original_movie_id = self.movie_decoder[encoded_movie_id]
            if original_movie_id != movie_id:
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((original_movie_id, float(similarity)))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_model(self, filepath: str):
        """Save the complete model and encoders."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save the Keras model (using new .keras format)
        self.model.save(f"{filepath}_model.keras")
        
        # Save encoders and metadata
        metadata = {
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder,
            'user_decoder': self.user_decoder,
            'movie_decoder': self.movie_decoder,
            'genre_encoder': self.genre_encoder,
            'year_normalizer': self.year_normalizer,
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model and encoders."""
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            num_users=metadata['num_users'],
            num_movies=metadata['num_movies'],
            embedding_dim=metadata['embedding_dim'],
            hidden_dims=metadata['hidden_dims'],
            dropout_rate=metadata['dropout_rate'],
            l2_reg=metadata['l2_reg'],
            learning_rate=metadata['learning_rate']
        )
        
        # Restore encoders
        instance.user_encoder = metadata['user_encoder']
        instance.movie_encoder = metadata['movie_encoder']
        instance.user_decoder = metadata['user_decoder']
        instance.movie_decoder = metadata['movie_decoder']
        instance.genre_encoder = metadata['genre_encoder']
        instance.year_normalizer = metadata['year_normalizer']
        
        # Load Keras model (try .keras first, then .h5 for backward compatibility)
        keras_path = f"{filepath}_model.keras"
        h5_path = f"{filepath}_model.h5"
        
        if os.path.exists(keras_path):
            instance.model = keras.models.load_model(keras_path)
        elif os.path.exists(h5_path):
            instance.model = keras.models.load_model(h5_path)
            logger.warning("Loaded legacy .h5 model file. Consider re-saving in .keras format.")
        else:
            raise FileNotFoundError(f"No model file found at {filepath}")
        
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance


if __name__ == "__main__":
    # Quick test of the model architecture
    print("🧠 Advanced Neural Collaborative Filtering Model")
    print("================================================")
    print("Architecture: User/Movie Embeddings + Multi-layer Neural Network")
    print("Features: Genres, Release Year, Popularity, User Activity")
    print("Output: Predicted Rating (1-5 scale)")
    print("\nReady for training on your movie database! 🎬")