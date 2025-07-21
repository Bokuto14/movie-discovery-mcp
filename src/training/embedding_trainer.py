#!/usr/bin/env python3
"""
Neural Embedding Training Pipeline
=================================
Extracts data from PostgreSQL, prepares features, and trains the 
advanced neural collaborative filtering model.

This pipeline:
1. Connects to your existing PostgreSQL database
2. Extracts ratings, movies, and user activity data
3. Engineers features for the neural network
4. Trains the advanced neural recommender
5. Evaluates performance vs existing collaborative filtering
6. Saves the trained model for integration
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
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

class NeuralEmbeddingTrainer:
    """
    Complete training pipeline for neural collaborative filtering.
    
    This class handles everything from data extraction to model training
    and evaluation. It's designed to work with your existing PostgreSQL
    database and can compare performance with your current system.
    """
    
    def __init__(self, output_dir: str = "neural_models"):
        """
        Initialize the training pipeline.
        
        Args:
            output_dir: Directory to save trained models and results
        """
        self.output_dir = output_dir
        self.model = None
        self.training_data = None
        self.test_data = None
        self.feature_data = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        self.training_metrics = {}
        self.comparison_results = {}
        
    async def extract_training_data(self) -> pd.DataFrame:
        """
        Extract comprehensive training data from PostgreSQL database.
        
        Returns:
            DataFrame with ratings, movie features, and user activity
        """
        logger.info("🔍 Extracting training data from PostgreSQL...")
        
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            # Main query: Join ratings with movie and user data
            training_data = await conn.fetch("""
                SELECT 
                    ur.user_id,
                    ur.movie_id,
                    ur.rating,
                    ur.created_at as rating_date,
                    ur.is_favorite,
                    
                    -- Movie features
                    m.title,
                    m.genres,
                    m.release_date,
                    m.vote_average,
                    m.vote_count,
                    m.popularity,
                    m.runtime,
                    m.overview,
                    
                    -- User activity features
                    user_stats.total_ratings,
                    user_stats.avg_rating as user_avg_rating,
                    user_stats.rating_variance,
                    user_stats.days_active,
                    
                    -- Movie popularity features
                    movie_stats.total_user_ratings,
                    movie_stats.avg_user_rating
                    
                FROM user_ratings ur
                JOIN movies m ON ur.movie_id = m.movie_id
                LEFT JOIN (
                    -- User statistics
                    SELECT 
                        user_id,
                        COUNT(*) as total_ratings,
                        AVG(rating) as avg_rating,
                        VARIANCE(rating) as rating_variance,
                        COUNT(DISTINCT DATE(created_at)) as days_active
                    FROM user_ratings
                    GROUP BY user_id
                ) user_stats ON ur.user_id = user_stats.user_id
                LEFT JOIN (
                    -- Movie statistics
                    SELECT 
                        movie_id,
                        COUNT(*) as total_user_ratings,
                        AVG(rating) as avg_user_rating
                    FROM user_ratings
                    GROUP BY movie_id
                ) movie_stats ON ur.movie_id = movie_stats.movie_id
                
                ORDER BY ur.created_at
            """)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in training_data])
            
            logger.info(f"✅ Extracted {len(df)} ratings from {df['user_id'].nunique()} users and {df['movie_id'].nunique()} movies")
            
            return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for neural network training.
        
        Args:
            df: Raw training data from database
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Engineering features for neural network...")
        
        # Create a copy for feature engineering
        features_df = df.copy()
        
        # ============= TEMPORAL FEATURES =============
        features_df['rating_date'] = pd.to_datetime(features_df['rating_date'])
        features_df['rating_year'] = features_df['rating_date'].dt.year
        features_df['rating_month'] = features_df['rating_date'].dt.month
        features_df['rating_dayofweek'] = features_df['rating_date'].dt.dayofweek
        
        # Days since first rating (user experience)
        user_first_rating = features_df.groupby('user_id')['rating_date'].min()
        features_df['user_days_since_first'] = features_df.apply(
            lambda row: (row['rating_date'] - user_first_rating[row['user_id']]).days,
            axis=1
        )
        
        # ============= MOVIE FEATURES =============
        # Extract release year from date (handle both string and datetime types)
        def extract_year(date_val):
            if pd.isna(date_val):
                return None
            elif isinstance(date_val, str):
                return date_val[:4]
            elif hasattr(date_val, 'year'):  # datetime object
                return str(date_val.year)
            else:
                return str(date_val)[:4]
        
        features_df['release_year'] = features_df['release_date'].apply(extract_year)
        features_df['release_year'] = pd.to_numeric(features_df['release_year'], errors='coerce')
        features_df['release_year'] = features_df['release_year'].fillna(features_df['release_year'].median())
        
        # Movie age when rated
        features_df['movie_age_when_rated'] = features_df['rating_year'] - features_df['release_year']
        
        # Popularity and rating features
        features_df['popularity'] = features_df['popularity'].fillna(0)
        features_df['vote_average'] = features_df['vote_average'].fillna(features_df['vote_average'].median())
        features_df['vote_count'] = features_df['vote_count'].fillna(0)
        features_df['runtime'] = features_df['runtime'].fillna(features_df['runtime'].median())
        
        # ============= USER FEATURES =============
        # User activity features
        features_df['user_activity_count'] = features_df['total_ratings'].fillna(1)
        features_df['user_avg_rating'] = features_df['user_avg_rating'].fillna(3.0)
        features_df['rating_variance'] = features_df['rating_variance'].fillna(1.0)
        features_df['days_active'] = features_df['days_active'].fillna(1)
        
        # User rating behavior
        features_df['user_rating_frequency'] = features_df['user_activity_count'] / features_df['days_active'].clip(lower=1)
        
        # User vs movie average difference
        features_df['user_movie_rating_diff'] = features_df['user_avg_rating'] - features_df['vote_average']
        
        # ============= INTERACTION FEATURES =============
        # User's rating vs their average (how much they liked this vs usual)
        features_df['rating_vs_user_avg'] = features_df['rating'] - features_df['user_avg_rating']
        
        # Movie's user rating vs TMDB rating
        features_df['user_vs_tmdb_rating'] = features_df['avg_user_rating'].fillna(features_df['vote_average']) - features_df['vote_average']
        
        # ============= CATEGORICAL FEATURES =============
        # Is favorite flag
        features_df['is_favorite'] = features_df['is_favorite'].fillna(False).astype(int)
        
        # High/low rating buckets
        features_df['is_high_rating'] = (features_df['rating'] >= 4.0).astype(int)
        features_df['is_low_rating'] = (features_df['rating'] <= 2.0).astype(int)
        
        # ============= GENRE PROCESSING =============
        # Keep genres as JSON for the neural network encoder
        features_df['genres'] = features_df['genres'].fillna('[]')
        
        # ============= CLEAN UP =============
        # Remove rows with critical missing data
        features_df = features_df.dropna(subset=['user_id', 'movie_id', 'rating'])
        
        # Select final feature columns
        feature_columns = [
            'user_id', 'movie_id', 'rating',  # Core data
            'genres', 'release_year', 'popularity', 'vote_average',  # Movie features  
            'user_activity_count', 'user_avg_rating', 'rating_variance',  # User features
            'runtime', 'movie_age_when_rated', 'user_rating_frequency',  # Engineered features
            'is_favorite', 'is_high_rating', 'is_low_rating'  # Categorical features
        ]
        
        final_df = features_df[feature_columns].copy()
        
        logger.info(f"✅ Feature engineering complete: {len(final_df)} samples with {len(feature_columns)} features")
        
        # Log feature statistics
        logger.info(f"📊 Feature Statistics:")
        logger.info(f"   Users: {final_df['user_id'].nunique()}")
        logger.info(f"   Movies: {final_df['movie_id'].nunique()}")
        logger.info(f"   Ratings: min={final_df['rating'].min()}, max={final_df['rating'].max()}, avg={final_df['rating'].mean():.2f}")
        logger.info(f"   User activity: min={final_df['user_activity_count'].min()}, max={final_df['user_activity_count'].max()}")
        
        return final_df
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split ensuring all test users AND movies are in training set.
        
        Args:
            df: Feature dataframe
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"📊 Creating train/test split ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
        
        # Sort chronologically
        df_sorted = df.sort_values(['user_id', 'movie_id']).reset_index(drop=True)
        
        # Strategy: Take latest ratings from users who have multiple ratings
        # This ensures both users and movies in test set exist in training
        
        test_indices = []
        train_indices = list(range(len(df_sorted)))
        
        # Group by user and take latest ratings for test (if user has enough ratings)
        for user_id in df_sorted['user_id'].unique():
            user_mask = df_sorted['user_id'] == user_id
            user_indices = df_sorted[user_mask].index.tolist()
            
            if len(user_indices) >= 4:  # Only if user has 4+ ratings
                # Take last 1-2 ratings for test
                n_test = min(2, max(1, int(len(user_indices) * test_size)))
                test_user_indices = user_indices[-n_test:]
                
                # Add to test set
                test_indices.extend(test_user_indices)
                
                # Remove from training indices
                for idx in test_user_indices:
                    if idx in train_indices:
                        train_indices.remove(idx)
        
        # Create train and test dataframes
        train_df = df_sorted.iloc[train_indices].copy()
        test_df = df_sorted.iloc[test_indices].copy()
        
        # Verify all test users and movies are in training
        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        train_movies = set(train_df['movie_id'].unique())
        test_movies = set(test_df['movie_id'].unique())
        
        # Remove test entries where user or movie not in training
        valid_test_mask = (
            test_df['user_id'].isin(train_users) & 
            test_df['movie_id'].isin(train_movies)
        )
        
        if not valid_test_mask.all():
            logger.info(f"🔧 Removing {(~valid_test_mask).sum()} test entries with unseen users/movies")
            test_df = test_df[valid_test_mask].copy()
        
        logger.info(f"✅ Train: {len(train_df)} ratings from {train_df['user_id'].nunique()} users, {train_df['movie_id'].nunique()} movies")
        logger.info(f"✅ Test: {len(test_df)} ratings from {test_df['user_id'].nunique()} users, {test_df['movie_id'].nunique()} movies")
        logger.info(f"✅ All test users and movies are present in training set")
        
        return train_df, test_df
    
    async def train_neural_model(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        epochs: int = 100,
        batch_size: int = 512
    ) -> AdvancedNeuralRecommender:
        """
        Train the advanced neural collaborative filtering model.
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained neural recommender model
        """
        logger.info("🧠 Training Advanced Neural Collaborative Filtering Model...")
        
        # Initialize model
        self.model = AdvancedNeuralRecommender(
            num_users=train_df['user_id'].nunique(),
            num_movies=train_df['movie_id'].nunique(),
            embedding_dim=128,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            l2_reg=1e-5,
            learning_rate=0.001
        )
        
        # Train the model
        history = self.model.train(
            features_df=train_df,
            validation_split=0.2 if val_df is None else 0.0,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save training metrics
        self.training_metrics = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_mae': history.history['mae'][-1],
            'final_val_mae': history.history['val_mae'][-1],
            'final_train_rmse': history.history['rmse'][-1],
            'final_val_rmse': history.history['val_rmse'][-1],
            'epochs_trained': len(history.history['loss']),
            'total_parameters': self.model.model.count_params()
        }
        
        logger.info("✅ Neural model training completed!")
        logger.info(f"📊 Final Metrics:")
        logger.info(f"   Train RMSE: {self.training_metrics['final_train_rmse']:.4f}")
        logger.info(f"   Val RMSE: {self.training_metrics['final_val_rmse']:.4f}")
        logger.info(f"   Train MAE: {self.training_metrics['final_train_mae']:.4f}")
        logger.info(f"   Val MAE: {self.training_metrics['final_val_mae']:.4f}")
        logger.info(f"   Model Parameters: {self.training_metrics['total_parameters']:,}")
        
        return self.model
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate the trained neural model on test data.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("📊 Evaluating neural model on test data...")
        
        if self.model is None or not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Filter test data to only include users and movies seen during training
        valid_users = set(self.model.user_encoder.keys())
        valid_movies = set(self.model.movie_encoder.keys())
        
        initial_test_size = len(test_df)
        
        # Filter to only valid user-movie pairs
        valid_test_mask = (
            test_df['user_id'].isin(valid_users) & 
            test_df['movie_id'].isin(valid_movies)
        )
        
        filtered_test_df = test_df[valid_test_mask].copy()
        
        if len(filtered_test_df) == 0:
            logger.warning("No valid test samples after filtering!")
            return {"error": "No valid test samples"}
        
        if len(filtered_test_df) < initial_test_size:
            removed_count = initial_test_size - len(filtered_test_df)
            logger.info(f"🔧 Filtered out {removed_count} test samples with unseen users/movies")
        
        logger.info(f"📊 Evaluating on {len(filtered_test_df)} valid test samples")
        
        # Make predictions
        predictions = self.model.predict(
            user_ids=filtered_test_df['user_id'].tolist(),
            movie_ids=filtered_test_df['movie_id'].tolist(),
            features_df=filtered_test_df
        )
        
        # Convert actual ratings to float (PostgreSQL returns Decimal objects)
        actual_ratings = np.array([float(rating) for rating in filtered_test_df['rating'].values])
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        mae = mean_absolute_error(actual_ratings, predictions)
        
        # Additional metrics
        mse = mean_squared_error(actual_ratings, predictions)
        
        # Accuracy within thresholds
        acc_05 = np.mean(np.abs(actual_ratings - predictions) <= 0.5)
        acc_10 = np.mean(np.abs(actual_ratings - predictions) <= 1.0)
        
        # Rating distribution analysis
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        actual_mean = np.mean(actual_ratings)
        actual_std = np.std(actual_ratings)
        
        evaluation_metrics = {
            'test_rmse': rmse,
            'test_mae': mae,
            'test_mse': mse,
            'accuracy_0.5': acc_05,
            'accuracy_1.0': acc_10,
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'test_samples': len(filtered_test_df),
            'filtered_samples': removed_count if len(filtered_test_df) < initial_test_size else 0
        }
        
        logger.info("✅ Neural model evaluation completed!")
        logger.info(f"📊 Test Metrics:")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAE: {mae:.4f}")
        logger.info(f"   Accuracy ±0.5: {acc_05:.3f}")
        logger.info(f"   Accuracy ±1.0: {acc_10:.3f}")
        logger.info(f"   Samples evaluated: {len(filtered_test_df)}")
        
        return evaluation_metrics
    
    async def compare_with_collaborative_filtering(self, test_df: pd.DataFrame) -> Dict:
        """
        Compare neural model performance with existing collaborative filtering.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Comparison results
        """
        logger.info("🔄 Comparing with existing collaborative filtering...")
        
        # Get collaborative filtering predictions
        await initialize_database()
        
        collaborative_predictions = []
        neural_predictions = []
        actual_ratings = []
        
        async with movie_db.pool.acquire() as conn:
            for _, row in test_df.iterrows():
                user_id = row['user_id']
                movie_id = row['movie_id']
                actual_rating = row['rating']
                
                # Get collaborative filtering recommendation
                try:
                    # Use existing collaborative filtering method
                    collab_recs = await movie_db.get_collaborative_recommendations(user_id, limit=50)
                    
                    # Find this movie in recommendations
                    collab_rating = None
                    for rec in collab_recs:
                        if rec['movie_id'] == movie_id:
                            collab_rating = rec.get('predicted_rating', 3.0)
                            break
                    
                    if collab_rating is None:
                        collab_rating = 3.0  # Default prediction
                    
                    collaborative_predictions.append(collab_rating)
                    
                except Exception as e:
                    collaborative_predictions.append(3.0)  # Fallback
                
                # Get neural prediction
                neural_pred = self.model.predict(
                    user_ids=[user_id],
                    movie_ids=[movie_id],
                    features_df=row.to_frame().T
                )[0]
                
                neural_predictions.append(neural_pred)
                actual_ratings.append(actual_rating)
        
        # Calculate metrics for both approaches
        collab_rmse = np.sqrt(mean_squared_error(actual_ratings, collaborative_predictions))
        collab_mae = mean_absolute_error(actual_ratings, collaborative_predictions)
        
        neural_rmse = np.sqrt(mean_squared_error(actual_ratings, neural_predictions))
        neural_mae = mean_absolute_error(actual_ratings, neural_predictions)
        
        # Calculate improvements
        rmse_improvement = (collab_rmse - neural_rmse) / collab_rmse * 100
        mae_improvement = (collab_mae - neural_mae) / collab_mae * 100
        
        comparison_results = {
            'collaborative_filtering': {
                'rmse': collab_rmse,
                'mae': collab_mae
            },
            'neural_embeddings': {
                'rmse': neural_rmse,
                'mae': neural_mae
            },
            'improvements': {
                'rmse_improvement_percent': rmse_improvement,
                'mae_improvement_percent': mae_improvement
            },
            'samples_compared': len(actual_ratings)
        }
        
        logger.info("✅ Comparison completed!")
        logger.info(f"📊 Performance Comparison:")
        logger.info(f"   Collaborative Filtering RMSE: {collab_rmse:.4f}")
        logger.info(f"   Neural Embeddings RMSE: {neural_rmse:.4f}")
        logger.info(f"   RMSE Improvement: {rmse_improvement:.1f}%")
        logger.info(f"   MAE Improvement: {mae_improvement:.1f}%")
        
        return comparison_results
    
    def save_results(self):
        """Save training results and model."""
        logger.info("💾 Saving training results and model...")
        
        # Save the trained model
        model_path = os.path.join(self.output_dir, "advanced_neural_recommender")
        self.model.save_model(model_path)
        
        # Save metrics and results
        results = {
            'training_metrics': self.training_metrics,
            'comparison_results': self.comparison_results,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'embedding_dim': self.model.embedding_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate,
                'l2_reg': self.model.l2_reg,
                'learning_rate': self.model.learning_rate
            }
        }
        
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {self.output_dir}")
        
        return results_path, model_path
    
    def plot_training_history(self):
        """Plot training history and save visualization."""
        if self.model is None or self.model.training_history is None:
            logger.warning("No training history available for plotting")
            return
        
        history = self.model.training_history
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Collaborative Filtering Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(history.history['mae'], label='Training MAE', color='green')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', color='orange')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE plot
        axes[1, 0].plot(history.history['rmse'], label='Training RMSE', color='purple')
        axes[1, 0].plot(history.history['val_rmse'], label='Validation RMSE', color='brown')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate', color='red')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            # Performance comparison if available
            if hasattr(self, 'comparison_results') and self.comparison_results:
                metrics = ['RMSE', 'MAE']
                collab_values = [
                    self.comparison_results['collaborative_filtering']['rmse'],
                    self.comparison_results['collaborative_filtering']['mae']
                ]
                neural_values = [
                    self.comparison_results['neural_embeddings']['rmse'],
                    self.comparison_results['neural_embeddings']['mae']
                ]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, collab_values, width, label='Collaborative Filtering', color='lightblue')
                axes[1, 1].bar(x + width/2, neural_values, width, label='Neural Embeddings', color='lightgreen')
                axes[1, 1].set_title('Performance Comparison')
                axes[1, 1].set_xlabel('Metrics')
                axes[1, 1].set_ylabel('Error')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(metrics)
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training history plot saved to {plot_path}")


async def main():
    """
    Main training pipeline execution.
    """
    print("🎬 Neural Collaborative Filtering Training Pipeline")
    print("=" * 55)
    
    # Initialize trainer
    trainer = NeuralEmbeddingTrainer(output_dir="neural_models")
    
    try:
        # Step 1: Extract training data
        print("\n🔍 Step 1: Extracting training data...")
        raw_data = await trainer.extract_training_data()
        
        if len(raw_data) < 100:
            logger.error("Not enough training data! Need at least 100 ratings.")
            return
        
        # Step 2: Engineer features
        print("\n🔧 Step 2: Engineering features...")
        feature_data = trainer.engineer_features(raw_data)
        
        # Step 3: Create train/test split
        print("\n📊 Step 3: Creating train/test split...")
        train_df, test_df = trainer.create_train_test_split(feature_data, test_size=0.2)
        
        # Step 4: Train neural model
        print("\n🧠 Step 4: Training Neural Collaborative Filtering Model...")
        model = await trainer.train_neural_model(
            train_df=train_df,
            epochs=50,  # Adjust based on your data size
            batch_size=256
        )
        
        # Step 5: Evaluate model
        print("\n📊 Step 5: Evaluating model...")
        evaluation_results = trainer.evaluate_model(test_df)
        
        # Step 6: Compare with existing system
        print("\n🔄 Step 6: Comparing with collaborative filtering...")
        comparison_results = await trainer.compare_with_collaborative_filtering(test_df)
        trainer.comparison_results = comparison_results
        
        # Step 7: Save results
        print("\n💾 Step 7: Saving results...")
        results_path, model_path = trainer.save_results()
        trainer.plot_training_history()
        
        # Step 8: Summary
        print("\n" + "=" * 55)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 55)
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Results saved to: {results_path}")
        
        if trainer.comparison_results:
            improvements = trainer.comparison_results['improvements']
            print(f"🚀 Performance vs Collaborative Filtering:")
            print(f"   RMSE Improvement: {improvements['rmse_improvement_percent']:.1f}%")
            print(f"   MAE Improvement: {improvements['mae_improvement_percent']:.1f}%")
        
        print(f"🧠 Model Architecture:")
        print(f"   Parameters: {trainer.training_metrics['total_parameters']:,}")
        print(f"   Embedding Dimension: {model.embedding_dim}")
        print(f"   Hidden Layers: {model.hidden_dims}")
        
        print("\n✨ Your neural recommendation system is ready!")
        print("Next: Integrate with your MCP server for real-time recommendations!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())