"""
Preprocess data for the recommendation engine
"""
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import logging

class DataPreprocessor:
    """Prepare user-movie interactions for training"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        
    def filter_sparse_users_items(self, df):
        """Remove users/items with too few interactions"""
        original_len = len(df)
        
        # Filter users with minimum interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        movie_counts = df['movie_id'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.config.min_interactions].index
        df = df[df['movie_id'].isin(valid_movies)]
        
        self.logger.info(f"Filtered: {original_len} -> {len(df)} interactions")
        self.logger.info(f"Users: {df['user_id'].nunique()}, Movies: {df['movie_id'].nunique()}")
        
        return df
    
    def create_mappings(self, df):
        """Create ID to index mappings"""
        unique_users = df['user_id'].unique()
        unique_movies = df['movie_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        self.logger.info(f"Created mappings: {len(self.user_id_map)} users, {len(self.movie_id_map)} movies")
    
    def build_interaction_matrix(self, df):
        """Build sparse user-movie interaction matrix"""
        # Map IDs to indices
        user_indices = df['user_id'].map(self.user_id_map).values
        movie_indices = df['movie_id'].map(self.movie_id_map).values
        
        # Apply confidence weighting
        ratings = df['rating'].values
        
        if ratings.max() <= 1:
            confidence = 1 + self.config.alpha * ratings
        else:
            confidence = 1 + self.config.alpha * np.log(1 + ratings / ratings.max())
        
        # Create sparse matrix
        n_users = len(self.user_id_map)
        n_movies = len(self.movie_id_map)
        
        interaction_matrix = csr_matrix(
            (confidence, (user_indices, movie_indices)),
            shape=(n_users, n_movies),
            dtype=np.float32
        )
        
        self.logger.info(f"Built matrix: {interaction_matrix.shape}, "
                        f"density: {interaction_matrix.nnz / (n_users * n_movies):.4%}")
        
        return interaction_matrix
    
    def temporal_train_test_split(self, df):
        """Split data temporally for realistic evaluation"""
        if 'timestamp' not in df.columns:
            raise ValueError("Temporal split requires 'timestamp' column")
        
        df = df.sort_values('timestamp')
        
        train_data = []
        test_data = []
        
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]
            n_test = max(1, int(len(user_df) * self.config.test_size))
            
            train_data.append(user_df.iloc[:-n_test])
            test_data.append(user_df.iloc[-n_test:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        self.logger.info(f"Temporal split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df