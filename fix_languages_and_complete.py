"""
Fix language detection and complete the dataset
Uses IMDb's actual language field + better detection
"""
import pandas as pd
import random
from datetime import datetime, timedelta

print("="*70)
print("🔧 FIXING LANGUAGE DETECTION & COMPLETING DATASET")
print("="*70)

# Load the raw IMDb data (it has originalLanguage field!)
print("\n[1/3] Loading IMDb raw data with language codes...")
basics_df = pd.read_csv('data/title_basics.tsv', sep='\t', na_values='\\N', low_memory=False)
ratings_df = pd.read_csv('data/title_ratings.tsv', sep='\t', na_values='\\N')

# Merge
merged = basics_df.merge(ratings_df, on='tconst', how='inner')

# Filter
filtered = merged[
    (merged['titleType'].isin(['movie', 'tvSeries', 'tvMovie', 'tvMiniSeries'])) &
    (merged['startYear'].notna()) &
    (merged['averageRating'].notna()) &
    (merged['averageRating'] >= 6.0) &
    (merged['numVotes'] >= 500)
].copy()

filtered['startYear'] = pd.to_numeric(filtered['startYear'], errors='coerce')
filtered = filtered[filtered['startYear'] >= 1970].copy()
filtered['startYear'] = filtered['startYear'].astype(int)

print(f"✅ Filtered to {len(filtered):,} titles")

# COMPREHENSIVE Language Mapping
LANGUAGE_CODES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'hi': 'Hindi', 'ar': 'Arabic', 'tr': 'Turkish', 'nl': 'Dutch', 'sv': 'Swedish',
    'pl': 'Polish', 'da': 'Danish', 'fi': 'Finnish', 'no': 'Norwegian', 'cs': 'Czech',
    'hu': 'Hungarian', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'he': 'Hebrew',
    'el': 'Greek', 'ro': 'Romanian', 'uk': 'Ukrainian', 'fa': 'Persian', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi', 'ur': 'Urdu', 'ml': 'Malayalam',
    'kn': 'Kannada', 'pa': 'Punjabi', 'gu': 'Gujarati', 'ms': 'Malay', 'af': 'Afrikaans',
    'sw': 'Swahili', 'am': 'Amharic', 'my': 'Burmese', 'km': 'Khmer', 'lo': 'Lao',
    'si': 'Sinhala', 'ne': 'Nepali', 'fil': 'Filipino', 'tl': 'Tagalog', 'ca': 'Catalan',
    'eu': 'Basque', 'gl': 'Galician', 'cy': 'Welsh', 'ga': 'Irish', 'is': 'Icelandic',
    'sq': 'Albanian', 'mk': 'Macedonian', 'bg': 'Bulgarian', 'sr': 'Serbian', 'hr': 'Croatian',
    'bs': 'Bosnian', 'sl': 'Slovenian', 'sk': 'Slovak', 'et': 'Estonian', 'lv': 'Latvian',
    'lt': 'Lithuanian', 'hy': 'Armenian', 'ka': 'Georgian', 'az': 'Azerbaijani', 'kk': 'Kazakh',
    'uz': 'Uzbek', 'mn': 'Mongolian', 'ps': 'Pashto', 'ku': 'Kurdish', 'sd': 'Sindhi',
    'so': 'Somali', 'ha': 'Hausa', 'yo': 'Yoruba', 'ig': 'Igbo', 'zu': 'Zulu', 'xh': 'Xhosa'
}

def enhanced_language_detection(row):
    """Use IMDb's originalLanguage field + character detection"""
    
    # First try IMDb's originalLanguage field
    if pd.notna(row.get('originalLanguage')):
        lang_code = str(row['originalLanguage']).lower()[:2]
        if lang_code in LANGUAGE_CODES:
            return LANGUAGE_CODES[lang_code]
    
    # Fallback: detect from title characters
    title = str(row.get('originalTitle', ''))
    
    # Japanese (Hiragana, Katakana, Kanji)
    if any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in title):
        return 'Japanese'
    # Korean (Hangul)
    elif any('\uAC00' <= c <= '\uD7AF' for c in title):
        return 'Korean'
    # Chinese (Hanzi)
    elif any('\u4E00' <= c <= '\u9FFF' for c in title):
        return 'Chinese'
    # Thai
    elif any('\u0E00' <= c <= '\u0E7F' for c in title):
        return 'Thai'
    # Arabic
    elif any('\u0600' <= c <= '\u06FF' for c in title):
        return 'Arabic'
    # Hebrew
    elif any('\u0590' <= c <= '\u05FF' for c in title):
        return 'Hebrew'
    # Cyrillic (Russian, Ukrainian, etc)
    elif any('\u0400' <= c <= '\u04FF' for c in title):
        return 'Russian'
    # Greek
    elif any('\u0370' <= c <= '\u03FF' for c in title):
        return 'Greek'
    # Devanagari (Hindi, Marathi, Nepali)
    elif any('\u0900' <= c <= '\u097F' for c in title):
        return 'Hindi'
    # Tamil
    elif any('\u0B80' <= c <= '\u0BFF' for c in title):
        return 'Tamil'
    # Telugu
    elif any('\u0C00' <= c <= '\u0C7F' for c in title):
        return 'Telugu'
    # Gujarati
    elif any('\u0A80' <= c <= '\u0AFF' for c in title):
        return 'Gujarati'
    # Bengali
    elif any('\u0980' <= c <= '\u09FF' for c in title):
        return 'Bengali'
    # Default to English
    else:
        return 'English'

print("\n[2/3] Detecting languages comprehensively...")
filtered['language'] = filtered.apply(enhanced_language_detection, axis=1)

print(f"✅ Detected {filtered['language'].nunique()} languages!")
print("\nLanguage Distribution:")
lang_counts = filtered['language'].value_counts()
for lang, count in lang_counts.head(20).items():
    print(f"   {lang}: {count:,}")

# Mood mapping
def map_genre_to_mood(genres):
    if pd.isna(genres):
        return 'Entertaining'
    
    mood_map = {
        'Action': ['Action-Packed', 'Thrilling'],
        'Adventure': ['Adventurous', 'Epic'],
        'Comedy': ['Fun', 'Light-Hearted'],
        'Drama': ['Emotional', 'Thought-Provoking'],
        'Horror': ['Dark', 'Scary'],
        'Romance': ['Romantic', 'Heartwarming'],
        'Thriller': ['Suspenseful', 'Intense'],
        'Sci-Fi': ['Mind-Bending', 'Futuristic'],
        'Fantasy': ['Magical', 'Whimsical'],
        'Mystery': ['Mysterious', 'Intriguing'],
        'Crime': ['Gritty', 'Dark'],
        'Animation': ['Fun', 'Whimsical'],
        'Family': ['Heartwarming', 'Uplifting'],
        'War': ['Intense', 'Heroic'],
        'Documentary': ['Educational', 'Informative']
    }
    
    moods = []
    for genre in str(genres).split(','):
        if genre in mood_map:
            moods.extend(mood_map[genre])
    
    return ','.join(list(set(moods))[:3]) if moods else 'Entertaining'

filtered['mood'] = filtered['genres'].apply(map_genre_to_mood)

# Clean up other fields
# Fix director field
if 'directors' in filtered.columns:
    filtered['director'] = filtered['directors'].fillna('Unknown').apply(
        lambda x: str(x).split(',')[0] if str(x) != 'Unknown' else 'Unknown'
    )
else:
    filtered['director'] = 'Unknown'
filtered['runtimeMinutes'] = pd.to_numeric(filtered['runtimeMinutes'], errors='coerce').fillna(100).astype(int)
filtered['endYear'] = pd.to_numeric(filtered['endYear'], errors='coerce')
filtered['isAdult'] = filtered['isAdult'].fillna(0).astype(int)

# Sort by popularity and take top titles
filtered['popularity'] = filtered['numVotes'] * filtered['averageRating']
filtered = filtered.sort_values('popularity', ascending=False).head(50000)

# Save movies
movies_df = filtered[[
    'tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
    'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating',
    'numVotes', 'director', 'language', 'mood'
]].copy()

movies_df.to_csv('data/imdb_movies.csv', index=False)
print(f"\n✅ Saved {len(movies_df):,} movies with {movies_df['language'].nunique()} languages")

# Generate streaming data
print("\n[3/3] Generating streaming platforms...")

streaming_data = []
for _, movie in movies_df.iterrows():
    # Smart platform assignment based on genre and year
    genres = str(movie['genres']).split(',')
    year = movie['startYear']
    
    platforms = []
    
    # Netflix: Drama, Thriller, International
    if any(g in ['Drama', 'Thriller', 'Crime'] for g in genres) or year >= 2015:
        if random.random() < 0.35:
            platforms.append('netflix')
    
    # Prime Video: Action, Sci-Fi
    if any(g in ['Action', 'Sci-Fi', 'Adventure'] for g in genres):
        if random.random() < 0.30:
            platforms.append('prime_video')
    
    # Disney+: Animation, Family
    if any(g in ['Animation', 'Family', 'Fantasy'] for g in genres):
        if random.random() < 0.50:
            platforms.append('disney_plus')
    
    # Hulu: Comedy, Horror
    if any(g in ['Comedy', 'Horror'] for g in genres):
        if random.random() < 0.25:
            platforms.append('hulu')
    
    # HBO Max: Drama, Fantasy
    if any(g in ['Drama', 'Fantasy', 'Crime'] for g in genres):
        if random.random() < 0.25:
            platforms.append('hbo_max')
    
    # Apple TV+: Recent high-quality
    if year >= 2019 and movie['averageRating'] >= 7.5:
        if random.random() < 0.20:
            platforms.append('apple_tv_plus')
    
    # Ensure some availability
    if not platforms and random.random() < 0.30:
        platforms.append(random.choice(['netflix', 'prime_video', 'hulu']))
    
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
print("✅ Saved streaming data")

# Generate user interactions
print("\nGenerating user interactions...")
n_users = 20000
n_interactions = 500000

interactions = []
movie_ids = movies_df['tconst'].tolist()

for i in range(n_interactions):
    if i % 100000 == 0 and i > 0:
        print(f"  Progress: {i:,}/{n_interactions:,}")
    
    interactions.append({
        'user_id': f'user_{random.randint(0, n_users-1)}',
        'movie_id': random.choice(movie_ids),
        'rating': random.randint(1, 5),
        'timestamp': int((datetime.now() - timedelta(days=random.randint(0, 1095))).timestamp())
    })

interactions_df = pd.DataFrame(interactions)
interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'])
interactions_df.to_csv('data/user_interactions.csv', index=False)
print(f"✅ Saved {len(interactions_df):,} interactions")

# Final summary
print("\n" + "="*70)
print("🎉 COMPLETE DATABASE READY!")
print("="*70)
print(f"\n📊 Final Statistics:")
print(f"   Total Content: {len(movies_df):,}")
print(f"   Movies: {len(movies_df[movies_df['titleType'] == 'movie']):,}")
print(f"   TV Series: {len(movies_df[movies_df['titleType'].str.contains('tv', case=False)]):,}")
print(f"   Languages: {movies_df['language'].nunique()}")
print(f"   Years: {int(movies_df['startYear'].min())} - {int(movies_df['startYear'].max())}")
print(f"   Users: {interactions_df['user_id'].nunique():,}")
print(f"   Interactions: {len(interactions_df):,}")

print("\n🌍 Languages Available:")
for lang in sorted(movies_df['language'].unique())[:30]:
    count = len(movies_df[movies_df['language'] == lang])
    print(f"   • {lang}: {count:,} titles")

print("\n🚀 Ready to run: streamlit run streamlit_app.py")
print("="*70)