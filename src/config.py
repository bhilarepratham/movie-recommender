from dataclasses import dataclass
from pathlib import Path

@dataclass
class RecommenderConfig:
    """Configuration for recommendation system"""
    
    # Model hyperparameters
    factors: int = 200
    regularization: float = 0.01
    iterations: int = 50
    alpha: float = 40
    
    # Data parameters
    min_interactions: int = 3
    test_size: float = 0.2
    random_state: int = 42
    
    # File paths
    data_path: str = 'data/'
    model_path: str = 'models/'
    log_path: str = 'logs/'
    
    imdb_file: str = 'data/imdb_movies.csv'
    streaming_file: str = 'data/streaming_platforms.csv'
    interactions_file: str = 'data/user_interactions.csv'
    
    # Streaming platforms
    streaming_platforms: list = None
    
    def __post_init__(self):
        if self.streaming_platforms is None:
            self.streaming_platforms = [
                'Netflix',
                'Amazon Prime',
                'Disney+',
                'Hulu',
                'HBO Max',
                'Apple TV+',
                'Paramount+',
                'Peacock'
            ]
        
        # Create directories
        Path(self.data_path).mkdir(exist_ok=True)
        Path(self.model_path).mkdir(exist_ok=True)
        Path(self.log_path).mkdir(exist_ok=True)

DEFAULT_CONFIG = RecommenderConfig()