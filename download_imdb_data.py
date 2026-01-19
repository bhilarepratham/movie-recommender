"""
Download COMPLETE IMDb Dataset
- ALL languages preserved
- 150,000+ titles (adjustable to millions)
- Better streaming accuracy using popularity patterns
"""
import pandas as pd
import gzip
import shutil
import urllib.request
import random
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
import csv
import time
import socket

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

print("="*70)
print("[MOVIE] DOWNLOADING COMPLETE IMDB DATABASE")
print("="*70)
print("[INFO] This will get 150,000+ titles in ALL languages")
print("[INFO] Estimated time: 15-20 minutes\n")

# IMDb Dataset URLs
IMDB_URLS = {
    'basics': 'https://datasets.imdbws.com/title.basics.tsv.gz',
    'ratings': 'https://datasets.imdbws.com/title.ratings.tsv.gz',
    'crew': 'https://datasets.imdbws.com/title.crew.tsv.gz',
    'akas': 'https://datasets.imdbws.com/title.akas.tsv.gz'
}

def download_and_extract(url, filename, max_retries=3):
    """Download and extract with retry logic + progress reporting"""
    for attempt in range(max_retries):
        try:
            print(f"[DOWNLOAD] Downloading {filename} (attempt {attempt + 1}/{max_retries})...")
            gz_path = f'data/{filename}.gz'
            tsv_path = f'data/{filename}.tsv'
            
            socket.setdefaulttimeout(300)

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = downloaded * 100 / total_size if total_size > 0 else 0
                bar_length = 40
                filled_length = int(bar_length * percent / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f'\r[DOWNLOAD] |{bar}| {percent:6.2f}%')
                sys.stdout.flush()

            urllib.request.urlretrieve(url, gz_path, show_progress)
            print("\n[INFO] Download complete.")

            print(f"[EXTRACT] Extracting {filename}... please wait.")
            file_size = os.path.getsize(gz_path)
            bytes_read = 0
            chunk_size = 1024 * 1024
            with gzip.open(gz_path, 'rb') as f_in, open(tsv_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    bytes_read += len(chunk)
                    percent = bytes_read * 100 / file_size
                    sys.stdout.write(f'\r[EXTRACT] {percent:6.2f}% complete')
                    sys.stdout.flush()
            print("\n[INFO] Extraction complete.")

            os.remove(gz_path)
            print(f"[SUCCESS] {filename} ready!\n")
            return tsv_path

        except socket.timeout:
            print(f"[ERROR] Attempt {attempt + 1} failed - TIMEOUT")
            if attempt < max_retries - 1:
                print("[WAIT] Waiting 15 seconds before retry...")
                time.sleep(15)
            else:
                raise
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("[WAIT] Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                raise

# Download datasets
print("\n[STEP 1/5] Downloading IMDb datasets...\n")
basics_path = download_and_extract(IMDB_URLS['basics'], 'title_basics')
ratings_path = download_and_extract(IMDB_URLS['ratings'], 'title_ratings')
crew_path = download_and_extract(IMDB_URLS['crew'], 'title_crew')
akas_path = download_and_extract(IMDB_URLS['akas'], 'title_akas')

# Load datasets
print("\n[STEP 2/5] Loading datasets...")
print("[INFO] Loading millions of records...\n")

basics_df = pd.read_csv(basics_path, sep='\t', low_memory=False, na_values='\\N')
ratings_df = pd.read_csv(ratings_path, sep='\t', low_memory=False, na_values='\\N')
crew_df = pd.read_csv(crew_path, sep='\t', low_memory=False, na_values='\\N')
akas_df = pd.read_csv(akas_path, sep='\t', low_memory=False, na_values='\\N')

print(f"[SUCCESS] Loaded {len(basics_df):,} titles")

# Merge datasets
print("\n[STEP 3/5] Merging and processing...")
try:
    merged_df = basics_df.merge(ratings_df, on='tconst', how='inner')
    print(f"[SUCCESS] Merged basics with ratings: {len(merged_df):,} titles")
    
    if 'directors' in crew_df.columns:
        merged_df = merged_df.merge(crew_df, on='tconst', how='left')
        print(f"[SUCCESS] Merged with crew data: {len(merged_df):,} titles")
    else:
        print("[WARNING] Crew data doesn't have 'directors' column, skipping merge")
        merged_df['directors'] = None
except Exception as e:
    print(f"[ERROR] Merge failed: {e}")
    raise

# Filter for quality content
movies_filter = (
    (merged_df['titleType'] == 'movie') &
    (merged_df['startYear'].notna()) &
    (merged_df['averageRating'].notna()) &
    (merged_df['averageRating'] >= 5.5) &
    (merged_df['numVotes'] >= 100)
)

tv_filter = (
    (merged_df['titleType'].isin(['tvSeries', 'tvMovie', 'tvMiniSeries'])) &
    (merged_df['startYear'].notna()) &
    (merged_df['averageRating'].notna()) &
    (merged_df['averageRating'] >= 4.5) &
    (merged_df['numVotes'] >= 50)
)

filtered_df = pd.concat([merged_df[movies_filter], merged_df[tv_filter]], ignore_index=True)

filtered_df['startYear'] = pd.to_numeric(filtered_df['startYear'], errors='coerce')
filtered_df = filtered_df[filtered_df['startYear'] >= 1960].copy()
filtered_df['startYear'] = filtered_df['startYear'].astype(int)

print(f"[SUCCESS] Filtered to {len(filtered_df):,} quality titles")
print(f"   - Movies: {len(filtered_df[filtered_df['titleType'] == 'movie']):,}")
print(f"   - TV Series: {len(filtered_df[filtered_df['titleType'].isin(['tvSeries', 'tvMovie', 'tvMiniSeries'])]):,}")

# Extract language information
print("\n[STEP 3B/5] Extracting language information...")
language_map = {}
if 'titleId' in akas_df.columns and 'language' in akas_df.columns:
    for _, row in akas_df.iterrows():
        tconst = row['titleId']
        lang = row['language']
        if pd.notna(lang) and tconst not in language_map:
            language_map[tconst] = str(lang).strip()
else:
    print("[WARNING] akas.tsv missing titleId or language column, using fallback")

LANGUAGE_CODE_MAP = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
    'ko': 'Korean', 'zh': 'Chinese', 'hi': 'Hindi', 'ar': 'Arabic',
    'tr': 'Turkish', 'nl': 'Dutch', 'sv': 'Swedish', 'pl': 'Polish',
    'da': 'Danish', 'fi': 'Finnish', 'no': 'Norwegian', 'cs': 'Czech',
    'hu': 'Hungarian', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian',
    'he': 'Hebrew', 'el': 'Greek', 'ro': 'Romanian', 'uk': 'Ukrainian',
    'fa': 'Persian', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'ur': 'Urdu', 'ml': 'Malayalam', 'kn': 'Kannada',
    'pa': 'Punjabi', 'gu': 'Gujarati', 'ms': 'Malay', 'af': 'Afrikaans',
    'sw': 'Swahili', 'am': 'Amharic', 'my': 'Burmese', 'km': 'Khmer',
    'lo': 'Lao', 'si': 'Sinhala', 'ne': 'Nepali', 'fil': 'Filipino',
    'ca': 'Catalan', 'eu': 'Basque', 'gl': 'Galician', 'cy': 'Welsh',
    'ga': 'Irish', 'is': 'Icelandic', 'sq': 'Albanian', 'mk': 'Macedonian',
    'bg': 'Bulgarian', 'sr': 'Serbian', 'hr': 'Croatian', 'bs': 'Bosnian',
    'sl': 'Slovenian', 'sk': 'Slovak', 'et': 'Estonian', 'lv': 'Latvian',
    'lt': 'Lithuanian', 'hy': 'Armenian', 'ka': 'Georgian', 'az': 'Azerbaijani',
    'kk': 'Kazakh', 'uz': 'Uzbek', 'mn': 'Mongolian', 'ps': 'Pashto',
    'ku': 'Kurdish', 'sd': 'Sindhi', 'so': 'Somali', 'ha': 'Hausa',
    'yo': 'Yoruba', 'ig': 'Igbo', 'zu': 'Zulu', 'xh': 'Xhosa'
}

def get_language_from_akas(tconst):
    if tconst in language_map:
        lang_code = language_map[tconst]
        return LANGUAGE_CODE_MAP.get(lang_code, 'English')
    return 'English'

filtered_df['language'] = filtered_df['tconst'].apply(get_language_from_akas)
print(f"[SUCCESS] Extracted {filtered_df['language'].nunique()} unique languages")

# Enhanced mood mapping
def map_genre_to_mood(genres):
    try:
        if pd.isna(genres) or str(genres).strip() == '' or str(genres) == 'nan':
            return 'Entertaining'
        
        genre_list = [g.strip() for g in str(genres).split(',') if g.strip()]
        
        mood_map = {
            'Action': 'Action-Packed,Thrilling',
            'Adventure': 'Adventurous,Epic',
            'Comedy': 'Fun,Light-Hearted',
            'Drama': 'Emotional,Thought-Provoking',
            'Horror': 'Dark,Scary',
            'Romance': 'Romantic,Heartwarming',
            'Thriller': 'Suspenseful,Intense',
            'Sci-Fi': 'Mind-Bending',
            'Fantasy': 'Magical,Epic',
            'Mystery': 'Mysterious',
            'Crime': 'Intense,Gritty',
            'Animation': 'Fun,Whimsical',
            'Family': 'Heartwarming',
            'War': 'Intense,Heroic',
            'Documentary': 'Educational',
            'Biography': 'Inspiring',
            'Musical': 'Uplifting',
            'Western': 'Classic',
            'Sport': 'Inspiring'
        }
        
        moods = []
        for genre in genre_list:
            if genre in mood_map:
                moods.extend(mood_map[genre].split(','))
        
        unique_moods = list(set(moods))[:3]
        return ','.join(unique_moods) if unique_moods else 'Entertaining'
    except Exception as e:
        return 'Entertaining'

filtered_df['mood'] = filtered_df['genres'].apply(map_genre_to_mood)

# Handle directors
if 'directors' in filtered_df.columns:
    def get_director(directors):
        if pd.isna(directors) or directors == '\\N' or str(directors).strip() == '':
            return 'Unknown'
        return str(directors).split(',')[0].strip()
    filtered_df['director'] = filtered_df['directors'].apply(get_director)
else:
    filtered_df['director'] = 'Unknown'
    print("[WARNING] Directors column not found in merged data")

# Runtime and endYear
filtered_df['runtimeMinutes'] = pd.to_numeric(filtered_df['runtimeMinutes'], errors='coerce').fillna(100).astype(int)
filtered_df['endYear'] = pd.to_numeric(filtered_df['endYear'], errors='coerce')

# Select top content by popularity
filtered_df['popularity_score'] = filtered_df['numVotes'] * filtered_df['averageRating']
filtered_df = filtered_df.sort_values('popularity_score', ascending=False).head(150000)

print(f"[SUCCESS] Selected top {len(filtered_df):,} titles by popularity")

# Create final dataset
movies_df = filtered_df[[
    'tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
    'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating',
    'numVotes', 'director', 'language', 'mood'
]].copy()

movies_df['isAdult'] = movies_df['isAdult'].fillna(0).astype(int)
movies_df.to_csv('data/imdb_movies.csv', index=False)
print(f"[SUCCESS] Saved {len(movies_df):,} items to data/imdb_movies.csv")

# Streaming platform assignment
print("\n[STEP 4/5] Generating REALISTIC streaming data...")
print("Using popularity patterns and genre preferences...\n")

PLATFORM_PREFERENCES = {
    'netflix': ['Drama', 'Thriller', 'Comedy', 'Crime', 'Documentary'],
    'prime_video': ['Action', 'Drama', 'Thriller', 'Sci-Fi'],
    'disney_plus': ['Animation', 'Family', 'Fantasy', 'Adventure'],
    'hulu': ['Comedy', 'Drama', 'Horror', 'Reality'],
    'hbo_max': ['Drama', 'Fantasy', 'Crime', 'Thriller'],
    'apple_tv_plus': ['Drama', 'Sci-Fi', 'Documentary', 'Thriller']
}

def assign_realistic_streaming(row):
    genres = str(row['genres']).split(',')
    year = row['startYear']
    popularity = row['numVotes']
    
    platforms_assigned = []
    
    for platform, preferred_genres in PLATFORM_PREFERENCES.items():
        probability = 0.15
        
        if any(genre in preferred_genres for genre in genres):
            probability += 0.25
        
        if year >= 2015:
            probability += 0.15
        elif year >= 2010:
            probability += 0.10
        
        if popularity > 100000:
            probability += 0.10
        
        if platform == 'disney_plus':
            if 'Animation' in genres or 'Family' in genres:
                probability = 0.6
            else:
                probability = 0.05
        
        if random.random() < probability:
            platforms_assigned.append(platform)
    
    if not platforms_assigned and random.random() < 0.3:
        platforms_assigned.append(random.choice(list(PLATFORM_PREFERENCES.keys())))
    
    return platforms_assigned

streaming_data = []
for _, movie in movies_df.iterrows():
    platforms = assign_realistic_streaming(movie)
    
    streaming_data.append({
        'movie_id': movie['tconst'],
        'title': movie['primaryTitle'],
        'netflix': 1 if 'netflix' in platforms else 0,
        'prime_video': 1 if 'prime_video' in platforms else 0,
        'disney_plus': 1 if 'disney_plus' in platforms else 0,
        'hulu': 1 if 'hulu' in platforms else 0,
        'hbo_max': 1 if 'hbo_max' in platforms else 0,
        'apple_tv_plus': 1 if 'apple_tv_plus' in platforms else 0
    })

streaming_df = pd.DataFrame(streaming_data)
streaming_df.to_csv('data/streaming_platforms.csv', index=False)

coverage = {
    'Netflix': streaming_df['netflix'].sum(),
    'Amazon Prime': streaming_df['prime_video'].sum(),
    'Disney+': streaming_df['disney_plus'].sum(),
    'Hulu': streaming_df['hulu'].sum(),
    'HBO Max': streaming_df['hbo_max'].sum(),
    'Apple TV+': streaming_df['apple_tv_plus'].sum()
}

print("[SUCCESS] Streaming coverage (realistic distribution):")
for platform, count in coverage.items():
    percentage = (count / len(streaming_df)) * 100
    print(f"   {platform}: {count:,} titles ({percentage:.1f}%)")

# Generate user interactions
print("\n[STEP 5/5] Generating user interactions...")

n_users = 25000
target_interactions = 1500000

movie_ids = movies_df['tconst'].tolist()
movie_genre_dict = dict(zip(movies_df['tconst'], movies_df['genres']))
movie_lang_dict = dict(zip(movies_df['tconst'], movies_df['language']))

genre_to_movies = defaultdict(set)
for mid, genres in movie_genre_dict.items():
    if pd.isna(genres):
        continue
    for g in str(genres).split(','):
        gg = g.strip()
        if gg:
            genre_to_movies[gg].add(mid)

lang_to_movies = defaultdict(set)
for mid, lang in movie_lang_dict.items():
    if pd.isna(lang):
        continue
    lang_to_movies[lang].add(mid)

INDIAN_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Marathi', 'Bengali', 'Punjabi', 'Gujarati', 'Urdu']
indian_movies = set()
for lang in INDIAN_LANGUAGES:
    indian_movies.update(lang_to_movies.get(lang, set()))

print(f"[INFO] Found {len(indian_movies):,} Indian language movies")

user_preferences = {}
all_genres = list(genre_to_movies.keys())
all_languages = list(lang_to_movies.keys())

print(f"[INFO] Available genres: {len(all_genres)}, languages: {len(all_languages)}")

for user_id in range(n_users):
    gcount = random.randint(2, min(5, max(2, len(all_genres))))
    lcount = random.randint(1, min(4, max(1, len(all_languages))))
    
    selected_genres = random.sample(all_genres, min(gcount, len(all_genres)))
    selected_langs = random.sample(all_languages, min(lcount, len(all_languages)))
    
    is_indian_user = random.random() < 0.20
    
    user_preferences[f'user_{user_id}'] = {
        'genres': selected_genres,
        'languages': selected_langs,
        'is_indian_user': is_indian_user
    }

seen = set()
unique_count = 0
attempts = 0
max_attempts = target_interactions * 3

print(f"[PROGRESS] Generating {target_interactions:,} interactions from {n_users:,} users...\n")

with open('data/user_interactions.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id','movie_id','rating','timestamp'])
    
    while unique_count < target_interactions and attempts < max_attempts:
        attempts += 1
        user_id = f'user_{random.randint(0, n_users-1)}'
        prefs = user_preferences.get(user_id)
        
        if not prefs:
            continue
        
        movie_id = None
        
        if prefs.get('is_indian_user') and random.random() < 0.60 and indian_movies:
            movie_id = random.choice(tuple(indian_movies))
        elif random.random() < 0.65 and prefs.get('genres') and prefs.get('languages'):
            pref_g = random.choice(prefs['genres'])
            pref_l = random.choice(prefs['languages'])
            candidates = genre_to_movies.get(pref_g, set()) & lang_to_movies.get(pref_l, set())
            
            if not candidates:
                gen_candidates = set()
                for g in prefs['genres']:
                    gen_candidates |= genre_to_movies.get(g, set())
                lang_candidates = set()
                for l in prefs['languages']:
                    lang_candidates |= lang_to_movies.get(l, set())
                candidates = gen_candidates & lang_candidates
            
            if candidates:
                movie_id = random.choice(tuple(candidates))
        
        if not movie_id and movie_ids:
            movie_id = random.choice(movie_ids)
        
        if not movie_id:
            continue
        
        movie_genres = str(movie_genre_dict.get(movie_id, ''))
        movie_lang = movie_lang_dict.get(movie_id)
        
        is_match = any(g in movie_genres for g in prefs.get('genres', [])) and movie_lang in prefs.get('languages', [])
        
        if is_match:
            rating = random.choices([3, 4, 5], weights=[0.15, 0.35, 0.50])[0]
        else:
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.35, 0.25, 0.10])[0]
        
        timestamp = int((datetime.now() - timedelta(days=random.randint(0, 1095))).timestamp())
        
        key = (user_id, movie_id)
        if key in seen:
            continue
        
        seen.add(key)
        writer.writerow([user_id, movie_id, rating, timestamp])
        unique_count += 1
        
        if unique_count % 100000 == 0:
            print(f"[PROGRESS] Written {unique_count:,} interactions...")

users_count = len({k[0] for k in seen})
interactions_count = unique_count

# Create Indian language recommendations file
print("\n[STEP 5B/5] Creating Indian language recommendations...")

indian_movies_list = movies_df[movies_df['tconst'].isin(indian_movies)].copy()
indian_movies_list.to_csv('data/indian_movies.csv', index=False)

print(f"[SUCCESS] Saved {len(indian_movies_list):,} Indian language titles to data/indian_movies.csv")

# Final Statistics
print("\n" + "="*70)
print("[SUCCESS] COMPLETE DATABASE SUCCESSFULLY CREATED!")
print("="*70)
print (f"\n[STATS] Your Comprehensive Database:")
print(f"   Total Titles: {len(movies_df):,}")
print(f"   Movies: {len(movies_df[movies_df['titleType'] == 'movie']):,}")
tv_series_count = len(movies_df[movies_df['titleType'].isin(['tvSeries', 'tvMovie', 'tvMiniSeries'])])
print(f"   TV Series: {tv_series_count:,}")
print(f"   Languages: {movies_df['language'].nunique()} (ALL preserved!)")
print(f"   Top Languages: {', '.join(movies_df['language'].value_counts().head(15).index.tolist())}")
print(f"   Genres: {len(set(','.join(movies_df['genres'].dropna()).split(',')))}")
print(f"   Year Range: {int(movies_df['startYear'].min())} - {int(movies_df['startYear'].max())}")
print(f"   Average Rating: {movies_df['averageRating'].mean():.2f}")
print(f"   Users: {users_count:,}")
print(f"   Total Interactions: {interactions_count:,}")

print("\n[INDIAN CONTENT] Indian Language Breakdown:")
print(f"   Total Indian Language Titles: {len(indian_movies_list):,}")
for lang in INDIAN_LANGUAGES:
    count = len(indian_movies_list[indian_movies_list['language'] == lang])
    if count > 0:
        print(f"   {lang}: {count:,}")

print("\n[STREAMING] Streaming Distribution:")
total_with_streaming = len(streaming_df[streaming_df[['netflix', 'prime_video', 'disney_plus', 'hulu', 'hbo_max', 'apple_tv_plus']].sum(axis=1) > 0])
print(f"   Content with streaming: {total_with_streaming:,} ({(total_with_streaming/len(streaming_df)*100):.1f}%)")

print("\n[INFO] Ready to run: streamlit run streamlit_app.py")
print("="*70)
print("="*70)