"""
Complete diagnostic of language data
"""
import pandas as pd

print("="*70)
print("🔍 COMPLETE LANGUAGE DIAGNOSTIC")
print("="*70)

# Check raw data file
print("\n[1] Checking imdb_movies.csv...")
try:
    df = pd.read_csv('data/imdb_movies.csv')
    print(f"✅ File exists: {len(df):,} rows")
    print(f"✅ Columns: {df.columns.tolist()}")
    
    if 'language' in df.columns:
        print(f"\n[2] Language column found!")
        print(f"   Unique languages: {df['language'].nunique()}")
        print(f"\n   Full language list:")
        lang_dist = df['language'].value_counts()
        for lang, count in lang_dist.items():
            pct = (count/len(df))*100
            print(f"   • {lang}: {count:,} titles ({pct:.1f}%)")
        
        # Check for Indian languages specifically
        print(f"\n[3] Indian language check:")
        indian_langs = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 'Bengali', 'Gujarati', 'Punjabi', 'Marathi', 'Urdu']
        for lang in indian_langs:
            count = len(df[df['language'] == lang])
            if count > 0:
                print(f"   ✅ {lang}: {count:,} titles")
            else:
                print(f"   ❌ {lang}: 0 titles")
    else:
        print("❌ ERROR: No 'language' column in data!")
        
except FileNotFoundError:
    print("❌ ERROR: data/imdb_movies.csv not found!")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Check Streamlit app
print(f"\n[4] Checking what Streamlit sees...")
try:
    import sys
    sys.path.insert(0, 'src')
    from data_loader import MovieDataLoader
    from config import RecommenderConfig
    
    config = RecommenderConfig()
    loader = MovieDataLoader(config)
    loader.load_all_data()
    
    print(f"✅ Streamlit loaded: {len(loader.movies_df):,} movies")
    print(f"✅ Languages in Streamlit: {loader.movies_df['language'].nunique()}")
    
    if loader.movies_df['language'].nunique() < 5:
        print(f"\n⚠️  WARNING: Streamlit only sees {loader.movies_df['language'].nunique()} languages!")
        print(f"   But CSV has {df['language'].nunique()} languages")
        print(f"   → This is a CACHE or LOADING issue")
    
except Exception as e:
    print(f"❌ Streamlit loading error: {e}")

print("\n" + "="*70)