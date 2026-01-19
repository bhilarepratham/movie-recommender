"""
Get truly multilingual content by sampling across languages
"""
import pandas as pd
import random
from datetime import datetime, timedelta

print("="*70)
print("🌍 CREATING TRULY MULTILINGUAL DATABASE")
print("="*70)

print("\n[1/3] Loading IMDb data with language codes...")
basics_df = pd.read_csv('data/title_basics.tsv', sep='\t', na_values='\\N', low_memory=False)
ratings_df = pd.read_csv('data/title_ratings.tsv', sep='\t', na_values='\\N')

# Merge
merged = basics_df.merge(ratings_df, on='tconst', how='inner')

# Filter quality content
filtered = merged[
    (merged['titleType'].isin(['movie', 'tvSeries'])) &
    (merged['startYear'].notna()) &
    (merged['averageRating'].notna()) &
    (merged['averageRating'] >= 6.0) &
    (merged['numVotes'] >= 200)  # Lower threshold to get more diverse content
].copy()

filtered['startYear'] = pd.to_numeric(filtered['startYear'], errors='coerce')
filtered = filtered[(filtered['startYear'] >= 1970) & (filtered['startYear'] <= 2024)].copy()

print(f"Total quality titles available: {len(filtered):,}")

# COMPREHENSIVE language mapping
LANGUAGE_MAP = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'hi': 'Hindi', 'ar': 'Arabic', 'tr': 'Turkish', 'nl': 'Dutch', 'sv': 'Swedish',
    'pl': 'Polish', 'da': 'Danish', 'fi': 'Finnish', 'no': 'Norwegian', 'cs': 'Czech',
    'hu': 'Hungarian', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'he': 'Hebrew',
    'el': 'Greek', 'ro': 'Romanian', 'uk': 'Ukrainian', 'fa': 'Persian', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi', 'ur': 'Urdu', 'ml': 'Malayalam',
    'kn': 'Kannada', 'pa': 'Punjabi', 'gu': 'Gujarati', 'ms': 'Malay', 'fil': 'Filipino',
    'tl': 'Tagalog', 'ca': 'Catalan', 'is': 'Icelandic', 'sq': 'Albanian', 'mk': 'Macedonian',
    'bg': 'Bulgarian', 'sr': 'Serbian', 'hr': 'Croatian', 'sl': 'Slovenian', 'sk': 'Slovak',
    'lt': 'Lithuanian', 'lv': 'Latvian', 'et': 'Estonian', 'hy': 'Armenian', 'ka': 'Georgian',
    'az': 'Azerbaijani', 'kk': 'Kazakh', 'uz': 'Uzbek', 'mn': 'Mongolian', 'ne': 'Nepali',
    'si': 'Sinhala', 'my': 'Burmese', 'km': 'Khmer', 'lo': 'Lao', 'am': 'Amharic',
    'sw': 'Swahili', 'yo': 'Yoruba', 'ha': 'Hausa', 'ig': 'Igbo', 'zu': 'Zulu'
}

# Map language using IMDb's originalLanguage field
def map_language(lang_code):
    if pd.isna(lang_code):
        return 'English'
    code = str(lang_code).lower().strip()[:2]
    return LANGUAGE_MAP.get(code, 'English')

filtered['language'] = filtered['originalLanguage'].apply(map_language)

print("\n[2/3] Sampling diverse content across languages...")

# Count what we have per language
lang_counts = filtered['language'].value_counts()
print(f"\nLanguages available in IMDb data:")
for lang, count in lang_counts.head(30).items():
    print(f"   {lang}: {count:,}")

# STRATEGIC SAMPLING: Get diverse content
# Target: 30% English, 70% other languages
target_per_language = {
    'English': 15000,
    'Spanish': 3000,
    'French': 2500,
    'German': 2000,
    'Japanese': 2500,
    'Korean': 2000,
    'Chinese': 2000,
    'Hindi': 2000,
    'Italian': 1500,
    'Portuguese': 1500,
    'Russian': 1500,
    'Turkish': 1000,
    'Thai': 1000,
    'Arabic': 1000,
}

sampled_content = []

for language, target_count in target_per_language.items():
    lang_data = filtered[filtered['language'] == language]
    
    if len(lang_data) > 0:
        # Sort by popularity within language
        lang_data = lang_data.copy()
        lang_data['popularity'] = lang_data['numVotes'] * lang_data['averageRating']
        lang_data = lang_data.sort_values('popularity', ascending=False)
        
        # Take up to target count
        sample_size = min(target_count, len(lang_data))
        sampled_content.append(lang_data.head(sample_size))
        print(f"   ✓ Sampled {sample_size:,} {language} titles")

# Combine all samples
movies_df = pd.concat(sampled_content, ignore_index=True)

print(f"\n✅ Total sampled: {len(movies_df):,} titles")
print(f"✅ Languages included: {movies_df['language'].nunique()}")

# Add remaining fields
def map_mood(genres):
    if pd.isna(genres):
        return 'Entertaining'
    
    mood_map = {
        'Action': 'Action-Packed', 'Comedy': 'Fun', 'Drama': 'Emotional',
        'Horror': 'Dark', 'Romance': 'Romantic', 'Thriller': 'Suspenseful',
        'Sci-Fi': 'Mind-Bending', 'Fantasy': 'Magical', 'Animation': 'Whimsical',
        'Crime': 'Gritty', 'Mystery': 'Mysterious', 'Adventure': 'Epic'
    }
    
    moods = []
    for genre in str(genres).split(','):
        if genre in mood_map:
            moods.append(mood_map[genre])
    
    return ','.join(list(set(moods))[:3]) if moods else 'Entertaining'

movies_df['mood'] = movies_df['genres'].apply(map_mood)
movies_df['director'] = 'Unknown'
movies_df['runtimeMinutes'] = pd.to_numeric(movies_df['runtimeMinutes'], errors='coerce').fillna(100).astype(int)
movies_df['endYear'] = pd.to_numeric(movies_df['endYear'], errors='coerce')
movies_df['isAdult'] = movies_df['isAdult'].fillna(0).astype(int)
movies_df['startYear'] = movies_df['startYear'].astype(int)

# Select columns
final_df = movies_df[[
    'tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
    'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating',
    'numVotes', 'director', 'language', 'mood'
]].copy()

final_df.to_csv('data/imdb_movies.csv', index=False)
print(f"\n✅ Saved multilingual database!")

# Generate streaming
print("\n[3/3] Generating streaming & interactions...")

streaming_data = []
for _, movie in final_df.iterrows():
    platforms = []
    
    # Language-based platform preferences
    lang = movie['language']
    
    if lang in ['Japanese', 'Korean', 'Chinese', 'Thai']:
        # Asian content often on Netflix
        if random.random() < 0.50: platforms.append('netflix')
    
    if lang in ['Spanish', 'Portuguese']:
        # Latin content on multiple platforms
        if random.random() < 0.40: platforms.append('netflix')
        if random.random() < 0.30: platforms.append('prime_video')
    
    if lang in ['French', 'German', 'Italian']:
        # European content
        if random.random() < 0.35: platforms.append('netflix')
        if random.random() < 0.25: platforms.append('prime_video')
    
    if lang == 'English':
        if random.random() < 0.35: platforms.append('netflix')
        if random.random() < 0.30: platforms.append('prime_video')
        if random.random() < 0.25: platforms.append('hulu')
        if 'Animation' in str(movie['genres']) and random.random() < 0.40:
            platforms.append('disney_plus')
    
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
        'hbo_max': 1 if random.random() < 0.20 else 0,
        'apple_tv_plus': 1 if random.random() < 0.15 else 0
    })

pd.DataFrame(streaming_data).to_csv('data/streaming_platforms.csv', index=False)

# Interactions
movie_ids = final_df['tconst'].tolist()
interactions = []

print("Generating 500K interactions...")
for i in range(500000):
    if i % 100000 == 0 and i > 0:
        print(f"  {i:,}/500,000")
    
    interactions.append({
        'user_id': f'user_{random.randint(0, 20000)}',
        'movie_id': random.choice(movie_ids),
        'rating': random.randint(1, 5),
        'timestamp': int((datetime.now() - timedelta(days=random.randint(0, 1095))).timestamp())
    })

interactions_df = pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'movie_id'])
interactions_df.to_csv('data/user_interactions.csv', index=False)

print("\n" + "="*70)
print("🎉 MULTILINGUAL DATABASE COMPLETE!")
print("="*70)

print(f"\n📊 Final Statistics:")
print(f"   Total Content: {len(final_df):,}")
print(f"   Languages: {final_df['language'].nunique()}")
print(f"   Interactions: {len(interactions_df):,}")

print(f"\n🌍 Language Distribution:")
for lang, count in final_df['language'].value_counts().items():
    percentage = (count / len(final_df)) * 100
    print(f"   {lang}: {count:,} ({percentage:.1f}%)")

print("\n🚀 Ready to run: streamlit run streamlit_app.py")
print("="*70)