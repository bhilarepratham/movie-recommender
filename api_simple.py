from flask import Flask, request, jsonify
import sys
sys.path.insert(0, 'src')

from config import RecommenderConfig
from data_loader import MovieDataLoader
from preprocessor import DataPreprocessor
from movie_recommender import MovieRecommendationEngine
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model at startup
print("Loading model...")
config = RecommenderConfig()
loader = MovieDataLoader(config)
loader.load_all_data()

preprocessor = DataPreprocessor(config)
train_df = loader.interactions_df
train_df = preprocessor.filter_sparse_users_items(train_df)
preprocessor.create_mappings(train_df)
matrix = preprocessor.build_interaction_matrix(train_df)

engine = MovieRecommendationEngine(config, preprocessor, loader)
engine.train(matrix, show_progress=False)
print("✓ Model ready!")

@app.route('/recommend/<user_id>')
def recommend(user_id):
    """Get recommendations: http://localhost:5000/recommend/user_0"""
    n = request.args.get('n', default=10, type=int)
    platform = request.args.get('platform', default=None, type=str)
    
    platforms = [platform] if platform else None
    recs = engine.recommend(user_id, n=n, filter_platforms=platforms)
    
    return jsonify({
        'user_id': user_id,
        'count': len(recs),
        'recommendations': recs
    })

@app.route('/similar/<movie_id>')
def similar(movie_id):
    """Find similar movies: http://localhost:5000/similar/tt0468569"""
    n = request.args.get('n', default=5, type=int)
    similar_movies = engine.similar_movies(movie_id, n=n)
    
    return jsonify({
        'movie_id': movie_id,
        'count': len(similar_movies),
        'similar_movies': similar_movies
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("API Server Running!")
    print("="*70)
    print("\nTry these URLs in your browser:")
    print("  http://localhost:5000/recommend/user_0")
    print("  http://localhost:5000/recommend/user_0?platform=Netflix")
    print("  http://localhost:5000/similar/tt0468569")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)