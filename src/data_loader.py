import pandas as pd
import logging

class MovieDataLoader:
    """Load and manage all data files"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.movies_df = None
        self.streaming_df = None
        self.interactions_df = None
    
    def load_all_data(self):
        """Load all data files"""
        self.load_imdb_data()
        self.load_streaming_data()
        self.load_interactions()
        return self
    
    def load_imdb_data(self, filepath=None):
        """Load IMDb movie data"""
        filepath = filepath or self.config.imdb_file
        self.logger.info(f"Loading IMDb data from {filepath}")
        
        try:
            self.movies_df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(self.movies_df)} movies")
            return self.movies_df
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            self.logger.info("Run: python src\\data_generator.py")
            raise
    
    def load_streaming_data(self, filepath=None):
        """Load streaming platform data"""
        filepath = filepath or self.config.streaming_file
        self.logger.info(f"Loading streaming data from {filepath}")
        
        try:
            self.streaming_df = pd.read_csv(filepath)
            self.logger.info(f"Loaded streaming data for {len(self.streaming_df)} movies")
            return self.streaming_df
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
    
    def load_interactions(self, filepath=None):
        """Load user interactions"""
        filepath = filepath or self.config.interactions_file
        self.logger.info(f"Loading interactions from {filepath}")
        
        try:
            self.interactions_df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(self.interactions_df):,} interactions")
            self.logger.info(f"  Users: {self.interactions_df['user_id'].nunique():,}")
            self.logger.info(f"  Movies: {self.interactions_df['movie_id'].nunique()}")
            return self.interactions_df
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
    
    def get_movie_details(self, movie_id):
        """Get full movie details including streaming"""
        movie_row = self.movies_df[self.movies_df['tconst'] == movie_id]
        
        if movie_row.empty:
            return None
        
        movie = movie_row.iloc[0]
        
        streaming_row = self.streaming_df[self.streaming_df['movie_id'] == movie_id]
        
        platforms = []
        if not streaming_row.empty:
            stream = streaming_row.iloc[0]
            platform_mapping = {
                'netflix': 'Netflix',
                'prime_video': 'Amazon Prime',
                'disney_plus': 'Disney+',
                'hulu': 'Hulu',
                'hbo_max': 'HBO Max',
                'apple_tv_plus': 'Apple TV+'
            }
            
            for key, name in platform_mapping.items():
                if key in stream and stream[key] == 1:
                    platforms.append(name)
        
        return {
            'movie_id': movie_id,
            'title': movie['primaryTitle'],
            'year': int(movie['startYear']) if pd.notna(movie['startYear']) else None,
            'genres': movie['genres'],
            'imdb_rating': float(movie['averageRating']) if pd.notna(movie['averageRating']) else None,
            'director': movie.get('director', 'Unknown'),
            'runtime': int(movie['runtimeMinutes']) if pd.notna(movie.get('runtimeMinutes')) else None,
            'streaming_platforms': platforms if platforms else ['Not Available']
        }
    
    def get_data_statistics(self):
        """Get dataset statistics"""
        stats = {
            'n_movies': len(self.movies_df),
            'n_interactions': len(self.interactions_df),
            'n_users': self.interactions_df['user_id'].nunique(),
            'n_rated_movies': self.interactions_df['movie_id'].nunique(),
            'avg_rating': self.interactions_df['rating'].mean(),
            'sparsity': 1 - (len(self.interactions_df) / 
                           (self.interactions_df['user_id'].nunique() * 
                            self.interactions_df['movie_id'].nunique()))
        }
        return stats