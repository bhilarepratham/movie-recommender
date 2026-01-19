"""
Enhanced Data Generator with Languages and TV Series
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Enhanced movie data with languages
movies_data = [
    {'tconst': 'tt0111161', 'titleType': 'movie', 'primaryTitle': 'The Shawshank Redemption', 'originalTitle': 'The Shawshank Redemption', 'isAdult': 0, 'startYear': 1994, 'endYear': None, 'runtimeMinutes': 142, 'genres': 'Drama', 'averageRating': 9.3, 'numVotes': 2500000, 'director': 'Frank Darabont', 'language': 'English', 'mood': 'Inspiring,Emotional'},
    {'tconst': 'tt0468569', 'titleType': 'movie', 'primaryTitle': 'The Dark Knight', 'originalTitle': 'The Dark Knight', 'isAdult': 0, 'startYear': 2008, 'endYear': None, 'runtimeMinutes': 152, 'genres': 'Action,Crime,Drama', 'averageRating': 9.0, 'numVotes': 2400000, 'director': 'Christopher Nolan', 'language': 'English', 'mood': 'Thrilling,Dark'},
    {'tconst': 'tt0137523', 'titleType': 'movie', 'primaryTitle': 'Fight Club', 'originalTitle': 'Fight Club', 'isAdult': 0, 'startYear': 1999, 'endYear': None, 'runtimeMinutes': 139, 'genres': 'Drama', 'averageRating': 8.8, 'numVotes': 1900000, 'director': 'David Fincher', 'language': 'English', 'mood': 'Dark,Intense'},
    {'tconst': 'tt1375666', 'titleType': 'movie', 'primaryTitle': 'Inception', 'originalTitle': 'Inception', 'isAdult': 0, 'startYear': 2010, 'endYear': None, 'runtimeMinutes': 148, 'genres': 'Action,Sci-Fi,Thriller', 'averageRating': 8.8, 'numVotes': 2200000, 'director': 'Christopher Nolan', 'language': 'English', 'mood': 'Mind-Bending,Thrilling'},
    {'tconst': 'tt0816692', 'titleType': 'movie', 'primaryTitle': 'Interstellar', 'originalTitle': 'Interstellar', 'isAdult': 0, 'startYear': 2014, 'endYear': None, 'runtimeMinutes': 169, 'genres': 'Adventure,Drama,Sci-Fi', 'averageRating': 8.7, 'numVotes': 1700000, 'director': 'Christopher Nolan', 'language': 'English', 'mood': 'Emotional,Epic'},
    {'tconst': 'tt0109830', 'titleType': 'movie', 'primaryTitle': 'Forrest Gump', 'originalTitle': 'Forrest Gump', 'isAdult': 0, 'startYear': 1994, 'endYear': None, 'runtimeMinutes': 142, 'genres': 'Drama,Romance', 'averageRating': 8.8, 'numVotes': 1900000, 'director': 'Robert Zemeckis', 'language': 'English', 'mood': 'Heartwarming,Inspiring'},
    {'tconst': 'tt0133093', 'titleType': 'movie', 'primaryTitle': 'The Matrix', 'originalTitle': 'The Matrix', 'isAdult': 0, 'startYear': 1999, 'endYear': None, 'runtimeMinutes': 136, 'genres': 'Action,Sci-Fi', 'averageRating': 8.7, 'numVotes': 1800000, 'director': 'Lana Wachowski', 'language': 'English', 'mood': 'Mind-Bending,Action-Packed'},
    {'tconst': 'tt0167260', 'titleType': 'movie', 'primaryTitle': 'The Lord of the Rings: The Return of the King', 'originalTitle': 'The Lord of the Rings: The Return of the King', 'isAdult': 0, 'startYear': 2003, 'endYear': None, 'runtimeMinutes': 201, 'genres': 'Adventure,Drama,Fantasy', 'averageRating': 9.0, 'numVotes': 1700000, 'director': 'Peter Jackson', 'language': 'English', 'mood': 'Epic,Heroic'},
    {'tconst': 'tt0110912', 'titleType': 'movie', 'primaryTitle': 'Pulp Fiction', 'originalTitle': 'Pulp Fiction', 'isAdult': 0, 'startYear': 1994, 'endYear': None, 'runtimeMinutes': 154, 'genres': 'Crime,Drama', 'averageRating': 8.9, 'numVotes': 1900000, 'director': 'Quentin Tarantino', 'language': 'English', 'mood': 'Quirky,Intense'},
    {'tconst': 'tt0073486', 'titleType': 'movie', 'primaryTitle': 'One Flew Over the Cuckoos Nest', 'originalTitle': 'One Flew Over the Cuckoos Nest', 'isAdult': 0, 'startYear': 1975, 'endYear': None, 'runtimeMinutes': 133, 'genres': 'Drama', 'averageRating': 8.7, 'numVotes': 950000, 'director': 'Milos Forman', 'language': 'English', 'mood': 'Thought-Provoking,Emotional'},
    {'tconst': 'tt0099685', 'titleType': 'movie', 'primaryTitle': 'Goodfellas', 'originalTitle': 'Goodfellas', 'isAdult': 0, 'startYear': 1990, 'endYear': None, 'runtimeMinutes': 145, 'genres': 'Biography,Crime,Drama', 'averageRating': 8.7, 'numVotes': 1100000, 'director': 'Martin Scorsese', 'language': 'English', 'mood': 'Intense,Gritty'},
    {'tconst': 'tt0245429', 'titleType': 'movie', 'primaryTitle': 'Spirited Away', 'originalTitle': 'Sen to Chihiro no kamikakushi', 'isAdult': 0, 'startYear': 2001, 'endYear': None, 'runtimeMinutes': 125, 'genres': 'Animation,Adventure,Family', 'averageRating': 8.6, 'numVotes': 720000, 'director': 'Hayao Miyazaki', 'language': 'Japanese', 'mood': 'Magical,Whimsical'},
    {'tconst': 'tt0317248', 'titleType': 'movie', 'primaryTitle': 'City of God', 'originalTitle': 'Cidade de Deus', 'isAdult': 0, 'startYear': 2002, 'endYear': None, 'runtimeMinutes': 130, 'genres': 'Crime,Drama', 'averageRating': 8.6, 'numVotes': 730000, 'director': 'Fernando Meirelles', 'language': 'Portuguese', 'mood': 'Intense,Gritty'},
    {'tconst': 'tt0076759', 'titleType': 'movie', 'primaryTitle': 'Star Wars: Episode IV - A New Hope', 'originalTitle': 'Star Wars', 'isAdult': 0, 'startYear': 1977, 'endYear': None, 'runtimeMinutes': 121, 'genres': 'Action,Adventure,Fantasy', 'averageRating': 8.6, 'numVotes': 1300000, 'director': 'George Lucas', 'language': 'English', 'mood': 'Epic,Adventurous'},
    {'tconst': 'tt0088763', 'titleType': 'movie', 'primaryTitle': 'Back to the Future', 'originalTitle': 'Back to the Future', 'isAdult': 0, 'startYear': 1985, 'endYear': None, 'runtimeMinutes': 116, 'genres': 'Adventure,Comedy,Sci-Fi', 'averageRating': 8.5, 'numVotes': 1100000, 'director': 'Robert Zemeckis', 'language': 'English', 'mood': 'Fun,Adventurous'},
    {'tconst': 'tt0482571', 'titleType': 'movie', 'primaryTitle': 'The Prestige', 'originalTitle': 'The Prestige', 'isAdult': 0, 'startYear': 2006, 'endYear': None, 'runtimeMinutes': 130, 'genres': 'Drama,Mystery,Sci-Fi', 'averageRating': 8.5, 'numVotes': 1200000, 'director': 'Christopher Nolan', 'language': 'English', 'mood': 'Mysterious,Thrilling'},
    {'tconst': 'tt0114369', 'titleType': 'movie', 'primaryTitle': 'Se7en', 'originalTitle': 'Seven', 'isAdult': 0, 'startYear': 1995, 'endYear': None, 'runtimeMinutes': 127, 'genres': 'Crime,Drama,Mystery', 'averageRating': 8.6, 'numVotes': 1500000, 'director': 'David Fincher', 'language': 'English', 'mood': 'Dark,Thrilling'},
    {'tconst': 'tt0034583', 'titleType': 'movie', 'primaryTitle': 'Casablanca', 'originalTitle': 'Casablanca', 'isAdult': 0, 'startYear': 1942, 'endYear': None, 'runtimeMinutes': 102, 'genres': 'Drama,Romance,War', 'averageRating': 8.5, 'numVotes': 550000, 'director': 'Michael Curtiz', 'language': 'English', 'mood': 'Romantic,Classic'},
    {'tconst': 'tt0047396', 'titleType': 'movie', 'primaryTitle': 'Rear Window', 'originalTitle': 'Rear Window', 'isAdult': 0, 'startYear': 1954, 'endYear': None, 'runtimeMinutes': 112, 'genres': 'Mystery,Thriller', 'averageRating': 8.5, 'numVotes': 480000, 'director': 'Alfred Hitchcock', 'language': 'English', 'mood': 'Suspenseful,Mysterious'},
    {'tconst': 'tt0103064', 'titleType': 'movie', 'primaryTitle': 'Terminator 2: Judgment Day', 'originalTitle': 'Terminator 2: Judgment Day', 'isAdult': 0, 'startYear': 1991, 'endYear': None, 'runtimeMinutes': 137, 'genres': 'Action,Sci-Fi', 'averageRating': 8.6, 'numVotes': 1050000, 'director': 'James Cameron', 'language': 'English', 'mood': 'Action-Packed,Thrilling'},
]

# Add TV Series
tv_series_data = [
    {'tconst': 'tt0944947', 'titleType': 'tvSeries', 'primaryTitle': 'Game of Thrones', 'originalTitle': 'Game of Thrones', 'isAdult': 0, 'startYear': 2011, 'endYear': 2019, 'runtimeMinutes': 57, 'genres': 'Action,Adventure,Drama', 'averageRating': 9.2, 'numVotes': 1900000, 'director': 'David Benioff', 'language': 'English', 'mood': 'Epic,Intense'},
    {'tconst': 'tt0903747', 'titleType': 'tvSeries', 'primaryTitle': 'Breaking Bad', 'originalTitle': 'Breaking Bad', 'isAdult': 0, 'startYear': 2008, 'endYear': 2013, 'runtimeMinutes': 49, 'genres': 'Crime,Drama,Thriller', 'averageRating': 9.5, 'numVotes': 1700000, 'director': 'Vince Gilligan', 'language': 'English', 'mood': 'Intense,Dark'},
    {'tconst': 'tt0185906', 'titleType': 'tvSeries', 'primaryTitle': 'Band of Brothers', 'originalTitle': 'Band of Brothers', 'isAdult': 0, 'startYear': 2001, 'endYear': 2001, 'runtimeMinutes': 60, 'genres': 'Drama,History,War', 'averageRating': 9.4, 'numVotes': 420000, 'director': 'Various', 'language': 'English', 'mood': 'Heroic,Emotional'},
    {'tconst': 'tt0141842', 'titleType': 'tvSeries', 'primaryTitle': 'The Sopranos', 'originalTitle': 'The Sopranos', 'isAdult': 0, 'startYear': 1999, 'endYear': 2007, 'runtimeMinutes': 55, 'genres': 'Crime,Drama', 'averageRating': 9.2, 'numVotes': 370000, 'director': 'David Chase', 'language': 'English', 'mood': 'Dark,Gritty'},
    {'tconst': 'tt0877057', 'titleType': 'tvSeries', 'primaryTitle': 'Death Note', 'originalTitle': 'Desu nôto', 'isAdult': 0, 'startYear': 2006, 'endYear': 2007, 'runtimeMinutes': 24, 'genres': 'Animation,Crime,Drama', 'averageRating': 9.0, 'numVotes': 310000, 'director': 'Tetsurô Araki', 'language': 'Japanese', 'mood': 'Mind-Bending,Dark'},
]

# Combine movies and TV series
all_content = movies_data + tv_series_data

print("Generating content dataset...")
content_df = pd.DataFrame(all_content)
content_df.to_csv('data/imdb_movies.csv', index=False)
print(f"✓ Saved {len(content_df)} items to data/imdb_movies.csv")

# Generate streaming data with realistic availability
print("Generating streaming data...")
streaming_data = []
for item in all_content:
    # Simulate realistic streaming patterns
    availability = {
        'movie_id': item['tconst'],
        'title': item['primaryTitle'],
        'netflix': 0,
        'prime_video': 0,
        'disney_plus': 0,
        'hulu': 0,
        'hbo_max': 0,
        'apple_tv_plus': 0
    }
    
    # Assign each content to 1-3 platforms
    num_platforms = random.randint(1, 3)
    platforms = random.sample(['netflix', 'prime_video', 'disney_plus', 'hulu', 'hbo_max', 'apple_tv_plus'], num_platforms)
    
    for platform in platforms:
        availability[platform] = 1
    
    streaming_data.append(availability)

streaming_df = pd.DataFrame(streaming_data)
streaming_df.to_csv('data/streaming_platforms.csv', index=False)
print(f"✓ Saved streaming data to data/streaming_platforms.csv")

# Generate user interactions
print("Generating user interactions...")
n_users = 5000
n_interactions = 300000
content_ids = [item['tconst'] for item in all_content]

interactions = []
for _ in range(n_interactions):
    interactions.append({
        'user_id': f'user_{random.randint(0, n_users-1)}',
        'movie_id': random.choice(content_ids),
        'rating': random.randint(1, 5),
        'timestamp': int((datetime.now() - timedelta(days=random.randint(0, 730))).timestamp())
    })

interactions_df = pd.DataFrame(interactions)
interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'])
interactions_df.to_csv('data/user_interactions.csv', index=False)
print(f"✓ Saved {len(interactions_df):,} interactions to data/user_interactions.csv")
print(f"  Users: {interactions_df['user_id'].nunique():,}")
print(f"  Content: {interactions_df['movie_id'].nunique()}")

print("\n✓ DATA GENERATION COMPLETE!")
print("Files created:")
print("  - data/imdb_movies.csv (movies + TV series)")
print("  - data/streaming_platforms.csv")
print("  - data/user_interactions.csv")