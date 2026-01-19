import sys
sys.path.insert(0, 'src')

from config import RecommenderConfig
from data_loader import MovieDataLoader
from preprocessor import DataPreprocessor
from movie_recommender import MovieRecommendationEngine
import logging

logging.basicConfig(level=logging.INFO)

# Load trained model
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

print("\n" + "="*70)
print("MOVIE RECOMMENDATION TESTER")
print("="*70)

# Interactive testing
while True:
    print("\n\nOptions:")
    print("1. Get recommendations for a user")
    print("2. Find similar movies")
    print("3. Filter by streaming platform")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        user_id = input("Enter user ID (e.g., user_0): ")
        n = int(input("How many recommendations? (default 10): ") or "10")
        
        recs = engine.recommend(user_id, n=n)
        
        print(f"\n🎬 Top {n} recommendations for {user_id}:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} ({rec['year']})")
            print(f"   ⭐ IMDb: {rec['imdb_rating']}")
            print(f"   🎭 Genres: {rec['genres']}")
            print(f"   📺 Streaming: {', '.join(rec['streaming_platforms'])}")
            print(f"   💯 Score: {rec['recommendation_score']}\n")
    
    elif choice == '2':
        movie_id = input("Enter movie ID (e.g., tt0468569 for The Dark Knight): ")
        n = int(input("How many similar movies? (default 5): ") or "5")
        
        similar = engine.similar_movies(movie_id, n=n)
        
        if similar:
            movie_details = loader.get_movie_details(movie_id)
            print(f"\n🎬 Movies similar to '{movie_details['title']}':\n")
            for i, movie in enumerate(similar, 1):
                print(f"{i}. {movie['title']} ({movie['year']})")
                print(f"   ⭐ IMDb: {movie['imdb_rating']}")
                print(f"   📺 {', '.join(movie['streaming_platforms'])}")
                print(f"   🔗 Similarity: {movie['similarity_score']}\n")
        else:
            print("Movie not found!")
    
    elif choice == '3':
        user_id = input("Enter user ID: ")
        print("\nAvailable platforms:")
        print("1. Netflix")
        print("2. Amazon Prime")
        print("3. Disney+")
        print("4. Hulu")
        print("5. HBO Max")
        
        platform_choice = input("\nEnter platform number: ")
        platforms_map = {
            '1': 'Netflix',
            '2': 'Amazon Prime',
            '3': 'Disney+',
            '4': 'Hulu',
            '5': 'HBO Max'
        }
        
        platform = platforms_map.get(platform_choice)
        if platform:
            recs = engine.recommend(user_id, n=10, filter_platforms=[platform])
            
            print(f"\n🎬 Recommendations on {platform}:\n")
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['title']} ({rec['year']}) - ⭐ {rec['imdb_rating']}")
        else:
            print("Invalid platform!")
    
    elif choice == '4':
        print("\nGoodbye! 🎬")
        break
    
    else:
        print("Invalid choice!")