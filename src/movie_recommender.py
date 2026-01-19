"""
Movie Recommendation Engine - Pure Python Version (No C++ required)
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import logging

class MovieRecommendationEngine:
    """Recommendation engine using SVD (pure Python)"""
    
    def __init__(self, config, preprocessor, data_loader):
        self.config = config
        self.preprocessor = preprocessor
        self.data_loader = data_loader
        self.model = None
        self.user_movie_matrix = None
        self.user_features = None
        self.movie_features = None
        self.logger = logging.getLogger(__name__)
        
    def train(self, interaction_matrix, show_progress=True):
        """Train the SVD model"""
        self.logger.info("Initializing SVD model...")
        self.user_movie_matrix = interaction_matrix
        
        # Use SVD instead of ALS (pure Python, no compilation needed)
        self.model = TruncatedSVD(
            n_components=min(self.config.factors, min(interaction_matrix.shape) - 1),
            random_state=self.config.random_state
        )
        
        self.logger.info(f"Training with SVD...")
        
        # Fit the model
        self.user_features = self.model.fit_transform(interaction_matrix)
        self.movie_features = self.model.components_.T
        
        self.logger.info("Training complete!")
        
    def recommend(self, user_id, n=10, filter_seen=True, filter_platforms=None):
        """Get movie recommendations"""
        if user_id not in self.preprocessor.user_id_map:
            self.logger.warning(f"User {user_id} not found, returning popular movies")
            return self._get_popular_movies(n, filter_platforms)
        
        user_idx = self.preprocessor.user_id_map[user_id]
        
        # Calculate scores for all movies
        user_vector = self.user_features[user_idx]
        scores = np.dot(self.movie_features, user_vector)
        
        # Filter already seen movies
        if filter_seen:
            seen_movies = self.user_movie_matrix[user_idx].nonzero()[1]
            scores[seen_movies] = -np.inf
        
        # Get top N movies
        top_movie_indices = np.argsort(scores)[::-1][:n*3 if filter_platforms else n]
        
        recommendations = []
        
        for movie_idx in top_movie_indices:
            movie_id = self.preprocessor.reverse_movie_map[movie_idx]
            
            # Get full movie details
            details = self.data_loader.get_movie_details(movie_id)
            
            if details is None:
                continue
            
            # Filter by streaming platform if requested
            if filter_platforms:
                available_platforms = set(details['streaming_platforms'])
                requested_platforms = set(filter_platforms)
                if not available_platforms.intersection(requested_platforms):
                    continue
            
            recommendations.append({
                'movie_id': movie_id,
                'title': details['title'],
                'year': details['year'],
                'genres': details['genres'],
                'imdb_rating': details['imdb_rating'],
                'director': details['director'],
                'runtime_minutes': details['runtime'],
                'streaming_platforms': details['streaming_platforms'],
                'recommendation_score': round(float(scores[movie_idx]), 3)
            })
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def similar_movies(self, movie_id, n=10):
        """Find similar movies"""
        if movie_id not in self.preprocessor.movie_id_map:
            self.logger.warning(f"Movie {movie_id} not found")
            return []
        
        movie_idx = self.preprocessor.movie_id_map[movie_id]
        
        # Calculate similarity with all movies
        movie_vector = self.movie_features[movie_idx]
        similarities = np.dot(self.movie_features, movie_vector)
        
        # Get top N similar movies (excluding itself)
        similarities[movie_idx] = -np.inf
        top_similar_indices = np.argsort(similarities)[::-1][:n]
        
        similar = []
        for idx in top_similar_indices:
            similar_movie_id = self.preprocessor.reverse_movie_map[idx]
            details = self.data_loader.get_movie_details(similar_movie_id)
            
            if details:
                similar.append({
                    'movie_id': similar_movie_id,
                    'title': details['title'],
                    'year': details['year'],
                    'genres': details['genres'],
                    'imdb_rating': details['imdb_rating'],
                    'streaming_platforms': details['streaming_platforms'],
                    'similarity_score': round(float(similarities[idx]), 3)
                })
        
        return similar
    
    def _get_popular_movies(self, n, filter_platforms=None):
        """Fallback: return popular movies for new users"""
        movies_df = self.data_loader.movies_df
        
        # Get top rated movies
        top_movies = movies_df.nlargest(n * 2, 'averageRating')
        
        recommendations = []
        
        for _, movie in top_movies.iterrows():
            details = self.data_loader.get_movie_details(movie['tconst'])
            
            if details is None:
                continue
            
            if filter_platforms:
                available_platforms = set(details['streaming_platforms'])
                if not available_platforms.intersection(set(filter_platforms)):
                    continue
            
            recommendations.append({
                'movie_id': movie['tconst'],
                'title': details['title'],
                'year': details['year'],
                'genres': details['genres'],
                'imdb_rating': details['imdb_rating'],
                'director': details['director'],
                'streaming_platforms': details['streaming_platforms'],
                'recommendation_score': float(details['imdb_rating'])
            })
            
            if len(recommendations) >= n:
                break
        
        return recommendations