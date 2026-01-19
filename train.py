"""
Complete training pipeline for movie recommendation engine
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

from config import RecommenderConfig
from data_loader import MovieDataLoader
from preprocessor import DataPreprocessor
from movie_recommender import MovieRecommendationEngine
from evaluator import RecommendationEvaluator

def setup_logging(config):
    """Setup logging"""
    log_file = Path(config.log_path) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    
    # Initialize
    config = RecommenderConfig()
    logger = setup_logging(config)
    
    print("\n" + "="*70)
    print("MOVIE RECOMMENDATION ENGINE - TRAINING")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("[STEP 1] Loading Data...")
    loader = MovieDataLoader(config)
    loader.load_all_data()
    
    stats = loader.get_data_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Movies: {stats['n_movies']}")
    print(f"  Users: {stats['n_users']}")
    print(f"  Interactions: {stats['n_interactions']:,}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Step 2: Preprocess
    print("\n[STEP 2] Preprocessing Data...")
    preprocessor = DataPreprocessor(config)
    
    train_df, test_df = preprocessor.temporal_train_test_split(loader.interactions_df)
    train_df = preprocessor.filter_sparse_users_items(train_df)
    preprocessor.create_mappings(train_df)
    interaction_matrix = preprocessor.build_interaction_matrix(train_df)
    
    # Step 3: Train
    print("\n[STEP 3] Training Model...")
    engine = MovieRecommendationEngine(config, preprocessor, loader)
    engine.train(interaction_matrix)
    
    # Step 4: Evaluate
    print("\n[STEP 4] Evaluating Model...")
    evaluator = RecommendationEvaluator()
    metrics = evaluator.evaluate_model(engine, test_df, k_values=[5, 10, 20])
    evaluator.print_results(metrics)
    
    # Check goal
    ctr = metrics['estimated_ctr_improvement_%']
    if ctr >= 17.0:
        print(f"SUCCESS! CTR improvement ({ctr:.2f}%) meets 17% target!")
    else:
        print(f"CTR improvement ({ctr:.2f}%) below target. Try hyperparameter tuning.")
    
    # Step 5: Demo
    print("\n[STEP 5] Sample Recommendations\n")
    print("="*70)
    
    sample_user = train_df['user_id'].iloc[0]
    
    # All platforms
    print(f"\nRecommendations for {sample_user}:")
    recs = engine.recommend(sample_user, n=5)
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec['title']} ({rec['year']})")
        print(f"   IMDb: {rec['imdb_rating']} | Genres: {rec['genres']}")
        print(f"   Streaming: {', '.join(rec['streaming_platforms'])}")
        print(f"   Score: {rec['recommendation_score']}")
    
    # Netflix only
    print(f"\n\nNetflix-Only Recommendations:")
    netflix_recs = engine.recommend(sample_user, n=3, filter_platforms=['Netflix'])
    for i, rec in enumerate(netflix_recs, 1):
        print(f"{i}. {rec['title']} ({rec['year']}) - IMDb: {rec['imdb_rating']}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    return engine, metrics

if __name__ == "__main__":
    engine, metrics = main()