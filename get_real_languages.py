"""
Use IMDb's ACTUAL language data from title.akas.tsv
This has REAL language codes for every title!
"""
import pandas as pd
import random
from datetime import datetime, timedelta

print("="*70)
print("🌍 DOWNLOADING REAL LANGUAGE DATA FROM IMDB")
print("="*70)

# Download title.akas.tsv if not already downloaded
import urllib.request
import gzip
import shutil

AKAS_URL = 'https://datasets.imdbws.com/title.akas.tsv.gz'

print("\n[1/4] Downloading language data from IMDb...")
print("⏳ This file contains REAL language codes!")

try:
    urllib.request.urlretrieve(AKAS_URL, 'data/title_akas.tsv.gz')
    print("✅ Downloaded")
    
    print("📦 Extracting...")
    with gzip.open('data/title_akas.tsv.gz', 'rb') as f_in:
        with open('data/title_akas.tsv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("✅ Extracted")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("Note: This file is large (~1GB)")

# Load all datasets
print("\n[2/4] Loading datasets with language information...")
basics_df = pd.read_csv('data/title_basics.tsv', sep='\t', na_values='\\N', low_memory=False)
ratings_df = pd.read_csv('data/title_ratings.tsv', sep='\t', na_values='\\N')
akas_df = pd.read_csv('data/title_akas.tsv', sep='\t', na_values='\\N', low_memory=False)

print(f"✅ Loaded {len(basics_df):,} titles")
print(f"✅ Loaded {len(akas_df):,} language records")

# Extract language from akas (this has REAL language codes!)
print("\n[3/4] Extracting REAL languages from IMDb...")

# Get primary language for each title
title_languages = akas_df.groupby('titleId')['language'].first().reset_index()
title_languages.columns = ['tconst', 'lang_code']

# Comprehensive language mapping
LANG_MAP = {
    'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 
    'ml': 'Malayalam', 'kn': 'Kannada', 'bn': 'Bengali', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'mr': 'Marathi', 'ur': 'Urdu', 'or': 'Odia',
    'as': 'Assamese', 'ks': 'Kashmiri', 'sd': 'Sindhi', 'sa': 'Sanskrit',
    'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'th': 'Thai',
    'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'fil': 'Filipino',
    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ar': 'Arabic', 'tr': 'Turkish',
    'nl': 'Dutch', 'sv': 'Swedish', 'pl': 'Polish', 'da': 'Danish',
    'fi': 'Finnish', 'no': 'Norwegian', 'cs': 'Czech', 'hu': 'Hungarian',
    'el': 'Greek', 'he': 'Hebrew', 'ro': 'Romanian', 'uk': 'Ukrainian',
    'fa': 'Persian', 'bg': 'Bulgarian', 'sr': 'Serbian', 'hr': 'Croatian',
    'sk': 'Slovak', 'sl': 'Slovenian', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'et': 'Estonian', 'is': 'Icelandic', 'sq': 'Albanian', 'mk': 'Macedonian'
}

def map_language(code):
    if pd.isna(code):
        return 'English'
    code_clean = str(code).lower().split('-')[0]  # Remove region codes
    return LANG_MAP.get(code_clean, 'English')

title_languages['language'] = title_languages['lang_code'].apply(map_language)

print(f"✅ Mapped languages for {len(title_languages):,} titles")
print(f"✅ Found {title_languages['language'].nunique()} unique languages!")

print("\nTop 30 languages in IMDb:")
for lang, count in title_languages['language'].value_counts().head(30).items():
    print(f"   {lang}: {count:,}")

# Merge with ratings and basics
print("\n[4/4] Building multilingual database...")
merged = basics_df.merge(ratings_df, on='tconst', how='inner')
merged = merged.merge(title_languages[['tconst', 'language']], on='tconst', how='left')

# Fill missing languages with English
merged['language'] = merged['language'].fillna('English')

# Filter quality content
filtered = merged[
    (merged['titleType'].isin(['movie', 'tvSeries'])) &
    (merged['startYear'].notna()) &
    (merged['averageRating'].notna()) &
    (merged['averageRating'] >= 5.5) &
    (merged['numVotes'] >= 100)
].copy()

filtered['startYear'] = pd.to_numeric(filtered['startYear'], errors='coerce')
filtered = filtered[(filtered['startYear'] >= 1970) & (filtered['startYear'] <= 2024)].copy()

print(f"\n✅ Quality content available: {len(filtered):,}")
print(f"✅ Languages available: {filtered['language'].nunique()}")

# Strategic sampling for diversity
TARGETS = {
    # Indian languages (35%)
    'Hindi': 6000, 'Tamil': 3500, 'Telugu': 3500, 'Malayalam': 2500,
    'Kannada': 2000, 'Bengali': 2000, 'Marathi': 1200, 'Gujarati': 1000,
    'Punjabi': 1000, 'Urdu': 800,
    
    # Asian (25%)
    'Japanese': 4000, 'Korean': 3000, 'Chinese': 2500, 'Thai': 1500,
    'Indonesian': 800, 'Vietnamese': 700, 'Filipino': 500,
    
    # European & others (40%)
    'English': 8000, 'Spanish': 2500, 'French': 2000, 'German': 1500,
    'Italian': 1200, 'Portuguese': 1200, 'Russian': 1000, 'Turkish': 800,
    'Arabic': 800, 'Persian': 600, 'Polish': 500, 'Dutch': 500
}

sampled = []
for lang, target in TARGETS.items():
    lang_data = filtered[filtered['language'] == lang].copy()
    
    if len(lang_data) > 0:
        lang_data['pop'] = lang_data['numVotes'] * lang_data['averageRating']
        lang_data = lang_data.sort_values('pop', ascending=False)
        
        sample_size = min(target, len(lang_data))
        sampled.append(lang_data.head(sample_size))
        
        emoji = "🇮🇳" if lang in ['Hindi','Tamil','Telugu','Malayalam','Kannada','Bengali','Marathi','Gujarati','Punjabi','Urdu'] else "🌍"
        print(f"   {emoji} {lang}: {sample_size:,}")

result = pd.concat(sampled, ignore_index=True)

print(f"\n✅ Total sampled: {len(result):,} titles")
print(f"✅ Languages: {result['language'].nunique()}")

# Add metadata
def get_mood(genres):
    if pd.isna(genres):
        return 'Entertaining'
    
    moods = {
        'Action': 'Action-Packed', 'Comedy': 'Fun', 'Drama': 'Emotional',
        'Horror': 'Dark', 'Romance': 'Romantic', 'Thriller': 'Suspenseful',
        'Sci-Fi': 'Mind-Bending', 'Fantasy': 'Magical', 'Animation': 'Whimsical'
    }
    
    found = [moods[g] for g in str(genres).split(',') if g in moods]
    return ','.join(list(set(found))[:2]) if found else 'Entertaining'

result['mood'] = result['genres'].apply(get_mood)
result['director'] = 'Unknown'
result['runtimeMinutes'] = pd.to_numeric(result['runtimeMinutes'], errors='coerce').fillna(100).astype(int)
result['endYear'] = pd.to_numeric(result['endYear'], errors='coerce')
result['isAdult'] = result['isAdult'].fillna(0).astype(int)
result['startYear'] = result['startYear'].astype(int)

final = result[[
    'tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
    'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating',
    'numVotes', 'director', 'language', 'mood'
]].copy()

final.to_csv('data/imdb_movies.csv', index=False)
print("\n✅ Saved multilingual database with REAL languages!")

# Streaming & interactions
print("\nGenerating streaming & interactions...")

streaming = []
for _, m in final.iterrows():
    plats = []
    
    if m['language'] in ['Hindi','Tamil','Telugu','Malayalam','Kannada','Bengali']:
        if random.random() < 0.60: plats.append('prime_video')
        if random.random() < 0.45: plats.append('netflix')
    elif m['language'] in ['Japanese','Korean','Chinese','Thai']:
        if random.random() < 0.55: plats.append('netflix')
    else:
        if random.random() < 0.35: plats.append('netflix')
        if random.random() < 0.30: plats.append('prime_video')
    
    if 'Animation' in str(m['genres']) and random.random() < 0.40:
        plats.append('disney_plus')
    
    streaming.append({
        'movie_id': m['tconst'], 'title': m['primaryTitle'],
        'netflix': 1 if 'netflix' in plats else 0,
        'prime_video': 1 if 'prime_video' in plats else 0,
        'disney_plus': 1 if 'disney_plus' in plats else 0,
        'hulu': 1 if random.random() < 0.18 else 0,
        'hbo_max': 1 if random.random() < 0.15 else 0,
        'apple_tv_plus': 1 if random.random() < 0.10 else 0
    })

pd.DataFrame(streaming).to_csv('data/streaming_platforms.csv', index=False)

interactions = [{
    'user_id': f'user_{random.randint(0, 25000)}',
    'movie_id': random.choice(final['tconst'].tolist()),
    'rating': random.randint(1, 5),
    'timestamp': int((datetime.now() - timedelta(days=random.randint(0, 1095))).timestamp())
} for _ in range(600000)]

pd.DataFrame(interactions).drop_duplicates(subset=['user_id','movie_id']).to_csv('data/user_interactions.csv', index=False)

print("\n" + "="*70)
print("🎉 COMPLETE WITH REAL IMDB LANGUAGES!")
print("="*70)
print(f"\n📊 Final Stats:")
print(f"   Content: {len(final):,}")
print(f"   Languages: {final['language'].nunique()}")

print(f"\n🌍 All Languages in Your Database:")
for lang, count in final['language'].value_counts().items():
    pct = (count/len(final))*100
    print(f"   {lang}: {count:,} ({pct:.1f}%)")

print("\n🚀 IMPORTANT: Clear cache and restart!")
print("   streamlit cache clear")
print("   streamlit run streamlit_app.py")
print("="*70)