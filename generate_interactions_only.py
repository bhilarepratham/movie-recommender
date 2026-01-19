"""
Generate ONLY user interactions - your movies are already saved!
"""
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

print("Generating user interactions...")

# Load your existing movie data
movies_df = pd.read_csv('data/imdb_movies.csv')
print(f"Found {len(movies_df):,} movies")

# Generate interactions - significantly increased for more data points
n_users = 50000  # Increased from 20k to 50k users
n_interactions = 2500000  # Increased from 800k to 2.5M interactions

movie_ids = movies_df['tconst'].tolist()

print("Generating interactions (this may take a few minutes)...")
print(f"Target: {n_interactions:,} interactions with {n_users:,} users")

# Pre-generate random values for better performance (vectorized)
print("  Generating random values...")
user_indices = np.random.randint(0, n_users, n_interactions)
movie_indices = np.random.randint(0, len(movie_ids), n_interactions)
ratings = np.random.randint(1, 6, n_interactions)  # 1-5 inclusive
days_ago = np.random.randint(0, 731, n_interactions)  # 0-730 days
base_time = int(datetime.now().timestamp())
timestamps = base_time - (days_ago * 86400)

# Build DataFrame directly (much faster than appending to list)
print("  Creating DataFrame...")
interactions_df = pd.DataFrame({
    'user_id': [f'user_{uid}' for uid in user_indices],
    'movie_id': [movie_ids[mid] for mid in movie_indices],
    'rating': ratings,
    'timestamp': timestamps
})
print("  Removing duplicates...")
interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'])

print("  Saving to CSV...")
interactions_df.to_csv('data/user_interactions.csv', index=False)

print(f"\n[SUCCESS] Saved {len(interactions_df):,} unique interactions")
print(f"[SUCCESS] Users: {interactions_df['user_id'].nunique():,}")
print(f"[SUCCESS] Movies: {interactions_df['movie_id'].nunique():,}")
print(f"[SUCCESS] Average interactions per user: {len(interactions_df) / interactions_df['user_id'].nunique():.1f}")
print("\n[INFO] All done! Run: streamlit run streamlit_app.py")