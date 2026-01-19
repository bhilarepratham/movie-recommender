"""
🎬 Complete Movie Recommendation Engine - Streamlit App
With working navigation, mood-based recommendations, and multilingual support
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="🎬 CineMatch", page_icon="🎬", layout="wide")

# TMDB Authentication Verification
def verify_tmdb_authentication():
    """Verify TMDB API authentication status"""
    url = "https://api.themoviedb.org/3/authentication"
    

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb_token']}"
    }

    
    try:
        response = requests.get(url, headers=headers)
        return response.text
    except Exception as e:
        return f"Authentication error: {str(e)}"

# Premium Dark Theme CSS
st.markdown("""
<style>
    :root {
        --primary: #FF6B6B;
        --secondary: #4ECDC4;
        --accent: #FFE66D;
    }
    
    * { margin: 0; padding: 0; }
    
    body, .stApp { 
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
        color: #e8e8e8 !important;
    }
    
    .header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2.5rem;
        border-radius: 18px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 50px rgba(255, 107, 107, 0.3);
    }
    
    .header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 0 3px 10px rgba(0,0,0,0.3);
    }
    
    .header p {
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    .movie-detail-card {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f1429 100%);
        border: 2px solid rgba(255, 107, 107, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    
    .movie-title-large {
        color: #FF6B6B;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .movie-rating-large {
        color: #FFE66D;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .meta-box {
        background: rgba(255, 107, 107, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #FF6B6B;
    }
    
    .meta-label {
        color: #FFE66D;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    .meta-value {
        color: #e8e8e8;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    
    .description-text {
        color: #9ca3af;
        line-height: 1.8;
        font-size: 1rem;
        margin-bottom: 2rem;
        background: rgba(255, 107, 107, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 3px solid #4ECDC4;
    }
    
    .mood-tag {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFB86B 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255,107,107,0.15) 0%, rgba(78,205,196,0.1) 100%);
        border: 2px solid rgba(255,107,107,0.3);
        padding: 1.3rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .stat-value {
        color: #FF6B6B;
        font-size: 2rem;
        font-weight: 800;
    }
    
    .stat-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .movie-detail-card, .header, .stat-card {
        animation: slideUp 0.6s ease-out;
    }
    
    .movie-detail-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.6);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_data():
    movies = pd.read_csv('data/imdb_movies.csv')
    streaming = pd.read_csv('data/streaming_platforms.csv')
    interactions = pd.read_csv('data/user_interactions.csv')
    try:
        indian_movies = pd.read_csv('data/indian_movies.csv')
    except:
        indian_movies = movies[movies['language'].isin(['Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam'])]
    return movies, streaming, interactions, indian_movies

movies_df, streaming_df, interactions_df, indian_movies_df = load_data()

# Enhanced movie details fetcher - Multiple sources
@st.cache_data(ttl=7200)
def get_movie_details(title, year, imdb_id=None, content_type='movie'):
    """Fetch movie details from multiple sources"""
    details = {
        'poster': None,
        'plot': None,
        'director': None,
        'actors': None,
        'runtime': None,
        'writer': None,
    }
    
    # First, try to get ID from search result if available
    poster_data = None
    tmdb_id = None
    
    # We need to find the TMDB ID first to get details and providers
    try:
        bearer_token = st.secrets["tmdb_token"]
        endpoint = "tv" if content_type and content_type.lower() in ['tv', 'tvseries', 'tvmovie'] else "movie"
        url = f"https://api.themoviedb.org/3/search/{endpoint}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {bearer_token}"}
        params = {
            "query": title,
            "year": int(year) if endpoint == "movie" else None,
            "first_air_date_year": int(year) if endpoint == "tv" else None,
            "language": "en-US"
        }
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            results = res.json().get('results', [])
            if results:
                tmdb_id = results[0]['id']
                poster_path = results[0].get('poster_path')
                if poster_path:
                    details['poster'] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                details['plot'] = results[0].get('overview') or details['plot']
    except:
        pass

    # If we have TMDB ID, get full details (credits for Director) and Watch Providers
    if tmdb_id:
        try:
            bearer_token = st.secrets["tmdb_token"]
            
            # Get Credits (Director)
            credits_url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/credits"
            headers = {"accept": "application/json", "Authorization": f"Bearer {bearer_token}"}
            credits_res = requests.get(credits_url, headers=headers, timeout=5)
            if credits_res.status_code == 200:
                credits_data = credits_res.json()
                # Find director in crew
                directors = [m['name'] for m in credits_data.get('crew', []) if m['job'] == 'Director']
                if directors:
                    details['director'] = ", ".join(directors[:2])
                
                # Get actors
                actors = [m['name'] for m in credits_data.get('cast', [])]
                if actors:
                    details['actors'] = ", ".join(actors[:4])

            # Get Watch Providers
            providers_url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/watch/providers"
            prov_res = requests.get(providers_url, headers=headers, timeout=5)
            if prov_res.status_code == 200:
                prov_data = prov_res.json()
                us_providers = prov_data.get('results', {}).get('US', {}).get('flatrate', [])
                if us_providers:
                    details['streaming'] = [p['provider_name'] for p in us_providers]
        except:
            pass

    # Fallback to OMDB if still missing info
    if not details['poster'] or not details['director']:
        omdb_data = get_omdb_data(title, year, imdb_id)
        if omdb_data:
            if not details['poster']: details['poster'] = omdb_data.get('poster')
            if not details['director']: details['director'] = omdb_data.get('director')
            if not details['plot'] or len(details['plot']) < 20: details['plot'] = omdb_data.get('plot')
            
    return details

# Fetch from TMDB using Bearer Token (authenticated)
def get_poster_from_tmdb_bearer(title, year, content_type='movie'):
    """Fetch poster from TMDB API using Bearer token authentication"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        # Determine endpoint based on content type
        endpoint = "tv" if content_type and content_type.lower() in ['tv', 'tvseries', 'tvmovie'] else "movie"
        url = f"https://api.themoviedb.org/3/search/{endpoint}"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "query": title,
            "year": int(year) if endpoint == "movie" else None, # TV search uses first_air_date_year usually, or just query
            "first_air_date_year": int(year) if endpoint == "tv" else None,
            "language": "en-US"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        # print(f"Error fetching poster for {title}: {e}")
        pass
    return None

# Get movie details by ID from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_movie_details(movie_id):
    """Fetch movie details from TMDB by ID"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "language": "en-US"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Get movie images from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_movie_images(movie_id):
    """Fetch movie images (posters, backdrops) from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Get trending movies from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_trending():
    """Fetch trending movies from TMDB discover endpoint"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = "https://api.themoviedb.org/3/discover/movie"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "include_adult": "false",
            "include_video": "false",
            "language": "en-US",
            "page": 1,
            "sort_by": "popularity.desc"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
    except Exception as e:
        pass
    return []

# Get trending TV series from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_trending_tv():
    """Fetch trending TV series from TMDB discover endpoint"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = "https://api.themoviedb.org/3/discover/tv"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "include_adult": "false",
            "include_null_first_air_dates": "false",
            "language": "en-US",
            "page": 1,
            "sort_by": "popularity.desc"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
    except Exception as e:
        pass
    return []

# Get daily trending content from TMDB (movies + TV)
@st.cache_data(ttl=1800)
def get_tmdb_daily_trending():
    """Fetch daily trending content (movies & TV) from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = "https://api.themoviedb.org/3/trending/all/day"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "language": "en-US"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
    except Exception as e:
        pass
    return []

# Search movie collections from TMDB
@st.cache_data(ttl=3600)
def search_tmdb_collections(query):
    """Search for movie collections from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = "https://api.themoviedb.org/3/search/collection"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
    except Exception as e:
        pass
    return []

# Get collection details from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_collection_details(collection_id):
    """Fetch collection details from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/collection/{collection_id}"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "language": "en-US"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Get collection images from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_collection_images(collection_id):
    """Fetch collection images from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/collection/{collection_id}/images"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Get watch providers from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_watch_providers(movie_id):
    """Fetch watch providers for a movie from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            # Return US providers by default, or just the whole results
            return data.get('results', {}).get('US', {})
    except Exception as e:
        pass
    return {}

# Get watch providers for TV series from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_tv_watch_providers(series_id):
    """Fetch watch providers for a TV series from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/tv/{series_id}/watch/providers"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            # Return US providers by default, or just the whole results
            return data.get('results', {}).get('US', {})
    except Exception as e:
        pass
    return {}

# Get TV series details and images from TMDB
@st.cache_data(ttl=3600)
def get_tmdb_tv_series_images(series_id):
    """Fetch TV series images from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/tv/{series_id}/images"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Get TV series details from TMDB (improved)
@st.cache_data(ttl=3600)
def get_tmdb_tv_series_details(series_id):
    """Fetch TV series details from TMDB"""
    try:
        bearer_token = st.secrets["tmdb_token"]
        
        url = f"https://api.themoviedb.org/3/tv/{series_id}"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        params = {
            "language": "en-US"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=8)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}

# Fetch from OMDB
def get_poster_from_omdb(title, year, imdb_id=None):
    """Fetch poster from OMDB"""
    try:
        api_key = st.secrets["omdb_key"]
        
        if imdb_id:
            url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        else:
            clean_title = title.replace(':', '').replace('?', '').replace('"', '').strip()
            url = f"http://www.omdbapi.com/?t={clean_title}&y={int(year)}&apikey={api_key}&type=movie"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True':
                poster = data.get('Poster')
                if poster and poster != 'N/A':
                    return poster
    except:
        pass
    return None

# Fetch OMDB data for plot, director, etc
def get_omdb_data(title, year, imdb_id=None):
    """Fetch plot, director, and other info from OMDB"""
    try:
        api_key = st.secrets["omdb_key"]
        
        if imdb_id:
            url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        else:
            clean_title = title.replace(':', '').replace('?', '').replace('"', '').strip()
            url = f"http://www.omdbapi.com/?t={clean_title}&y={int(year)}&apikey={api_key}&type=movie"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True':
                return {
                    'plot': data.get('Plot') if data.get('Plot') and data.get('Plot') != 'N/A' else None,
                    'director': data.get('Director') if data.get('Director') and data.get('Director') != 'N/A' else None,
                    'actors': data.get('Actors') if data.get('Actors') and data.get('Actors') != 'N/A' else None,
                    'runtime': data.get('Runtime') if data.get('Runtime') and data.get('Runtime') != 'N/A' else None,
                    'writer': data.get('Writer') if data.get('Writer') and data.get('Writer') != 'N/A' else None,
                }
    except:
        pass
    return {}

# Generate detailed plot description based on genres and mood
def generate_plot_description(movie):
    """Generate rich plot description from movie metadata"""
    # FIX: Handle NaN/float values for genres
    genres_str = movie['genres']
    if pd.isna(genres_str) or not isinstance(genres_str, str):
        genres_str = 'Entertainment'
    
    genres = [g.strip() for g in genres_str.split(',')][:2] if ',' in genres_str else [genres_str]
    mood = movie['mood']
    year = int(movie['startYear'])
    rating = movie['averageRating']
    
    templates = {
        'Drama': f"This compelling drama from {year} explores deep human emotions and relationships. With a rating of {rating}/10 and {int(movie['numVotes']):,} votes, it stands as a testament to powerful storytelling.",
        'Action': f"An action-packed thriller from {year} featuring intense sequences and thrilling moments. Rated {rating}/10, this film keeps audiences on the edge of their seats.",
        'Comedy': f"A hilarious {genres[0].lower()} from {year} that delivers laughs and entertainment. With {rating}/10 rating, audiences appreciate its humor and charm.",
        'Romance': f"A touching romantic tale from {year} exploring love and connection. Rated {rating}/10, it resonates with audiences seeking emotional engagement.",
        'Horror': f"A spine-chilling horror film from {year} designed to frighten and thrill. Rated {rating}/10, viewers rate this as a memorable scary experience.",
        'Sci-Fi': f"A mind-bending science fiction story from {year} exploring futuristic concepts. With {rating}/10 rating, it challenges viewers' imagination.",
        'Animation': f"A visually stunning animated feature from {year} for audiences of all ages. Rated {rating}/10, it combines artistry with engaging storytelling.",
        'Thriller': f"A suspenseful thriller from {year} filled with unexpected twists. Rated {rating}/10, this film keeps audiences guessing until the end.",
        'Adventure': f"An epic adventure from {year} taking audiences on an exciting journey. Rated {rating}/10, it delivers action and discovery.",
        'Mystery': f"An intriguing mystery from {year} that challenges viewers to solve puzzles. Rated {rating}/10, it offers engaging suspenseful entertainment."
    }
    
    for genre in genres:
        genre = genre.strip()
        if genre in templates:
            return templates[genre]
    
    return f"A captivating {mood.lower()} production from {year} featuring {', '.join(genres)}. Rated {rating}/10 by {int(movie['numVotes']):,} viewers, this film offers quality entertainment."

# Download image
@st.cache_data(ttl=3600)
def get_image(url):
    """Download image with error handling"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # st.write(f"DEBUG: Fetching {url}")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        # print(f"Error loading image {url}: {e}")
        pass
    return None

# Display movie with full details
def show_movie_full_detail(movie):
    """Display movie card with poster, description, and metadata"""
    
    st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
    
    col_poster, col_info = st.columns([1.1, 1.9])
    
    with col_poster:
        st.write("")
        details = get_movie_details(movie['primaryTitle'], movie['startYear'], movie.get('tconst'), movie.get('titleType'))
        
        if details and details.get('poster'):
            img = get_image(details['poster'])
            if img:
                st.image(img, use_container_width=True, output_format='JPEG')
            else:
                st.markdown(f"<div style='background:linear-gradient(135deg, #FF6B6B, #4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>🎬</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:linear-gradient(135deg, #FF6B6B, #4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📽️</div>", unsafe_allow_html=True)
    
    with col_info:
        st.markdown(f'<div class="movie-title-large">{movie["primaryTitle"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="movie-rating-large">⭐ {movie["averageRating"]}/10 <span style="font-size: 0.9rem; color: #9ca3af; font-weight: 400">({int(movie["numVotes"]):,} votes)</span></div>', unsafe_allow_html=True)
        
        # Metadata grid
        st.markdown('<div class="meta-grid">', unsafe_allow_html=True)
        
        # Year
        st.markdown(f'''
        <div class="meta-box">
            <div class="meta-label">📅 Year</div>
            <div class="meta-value">{movie['startYear']}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Type
        st.markdown(f'''
        <div class="meta-box">
            <div class="meta-label">🎬 Type</div>
            <div class="meta-value">{movie.get('titleType', 'Movie').replace('tv', 'TV ').title()}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Language
        st.markdown(f'''
        <div class="meta-box">
            <div class="meta-label">🌍 Language</div>
            <div class="meta-value">{movie['language']}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Runtime (from details)
        runtime = details.get('runtime', str(movie.get('runtimeMinutes', 'N/A')) + 'min') if details else str(movie.get('runtimeMinutes', 'N/A')) + 'min'
        st.markdown(f'''
        <div class="meta-box">
            <div class="meta-label">⏱ Runtime</div>
            <div class="meta-value">{runtime}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot Summary (Full from TMDB/OMDB if avail)
        plot = details.get('plot') if details else None
        if not plot:
            plot = f"A {movie['startYear']} {movie['language']} production. Rated {movie['averageRating']}/10."
            
        st.markdown("#### 📝 Plot Summary")
        if len(plot) > 280:
             st.markdown(f'<div class="description-text">{plot[:280]}...</div>', unsafe_allow_html=True)
             with st.expander("Read Full Plot"):
                 st.write(plot)
        else:
            st.markdown(f'<div class="description-text">{plot}</div>', unsafe_allow_html=True)

        # Streaming Providers display
        if details and details.get('streaming'):
            st.markdown("#### 📺 Where to Watch")
            provs = details['streaming']
            # Simple text badges for now
            badges = "".join([f'<span style="background: rgba(78,205,196,0.2); color: #4ECDC4; padding: 4px 10px; border-radius: 4px; margin-right: 8px; font-size: 0.9rem; border: 1px solid rgba(78,205,196,0.4)">{p}</span>' for p in provs])
            st.markdown(f'<div style="margin-bottom: 1.5rem">{badges}</div>', unsafe_allow_html=True)
            
        
        # Director & Cast
        c1, c2 = st.columns(2)
        with c1:
            director = details.get('director', 'N/A') if details else 'N/A'
            st.markdown(f"#### 👤 Director\n<div style='color: #e8e8e8; margin-bottom: 1rem'>{director}</div>", unsafe_allow_html=True)
            
        with c2:
            genres_raw = movie.get('genres')
            genres = str(genres_raw).replace(',', ', ') if isinstance(genres_raw, str) else 'Unknown'
            st.markdown(f"#### 🎭 Genres\n<div style='color: #e8e8e8; margin-bottom: 1rem'>{genres}</div>", unsafe_allow_html=True)
            
        if details and details.get('actors'):
            st.markdown(f"#### 👥 Cast\n<div style='color: #e8e8e8; margin-bottom: 1rem'>{details['actors']}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

# Display TV series with full details (improved)
def show_tv_series_detail(series_data):
    """Display TV series card with images and details"""
    
    st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
    
    col_poster, col_info = st.columns([1.1, 1.9])
    
    with col_poster:
        st.write("")
        # Try to get poster from TMDB
        if series_data.get('poster_path'):
            poster_url = f"https://image.tmdb.org/t/p/w500{series_data['poster_path']}"
            img = get_image(poster_url)
            if img:
                st.image(img, use_container_width=True, output_format='JPEG')
            else:
                st.markdown(f"<div style='background:linear-gradient(135deg, #FF6B6B, #4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📺</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:linear-gradient(135deg, #FF6B6B, #4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📺</div>", unsafe_allow_html=True)
    
    with col_info:
        title = series_data.get('name', 'Unknown')
        rating = series_data.get('vote_average', 0)
        vote_count = series_data.get('vote_count', 0)
        first_air_date = series_data.get('first_air_date', 'N/A')
        last_air_date = series_data.get('last_air_date', 'N/A')
        seasons = len(series_data.get('seasons', []))
        episodes = sum(len(s.get('episodes', [])) for s in series_data.get('seasons', []))
        
        st.markdown(f"""
        <div class="movie-title-large">{title}</div>
        <div class="movie-rating-large">⭐ {rating:.1f}/10 • {vote_count:,} votes</div>
        
        <div class="meta-grid">
            <div class="meta-box">
                <div class="meta-label">📅 First Aired</div>
                <div class="meta-value">{first_air_date}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">🏁 Last Aired</div>
                <div class="meta-value">{last_air_date}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">📺 Seasons</div>
                <div class="meta-value">{seasons}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">🎬 Episodes</div>
                <div class="meta-value">{episodes}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Plot description
    overview = series_data.get('overview', 'No description available')
    st.markdown(f"""
    <div class="description-text">
        <strong>📖 Overview:</strong><br/>
        {overview}
    </div>
    """, unsafe_allow_html=True)
    
    # Additional info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        networks = series_data.get('networks', [])
        if networks:
            network_names = ', '.join([n.get('name', '') for n in networks[:2]])
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:0.9rem;'>📡 Network</div><div style='color:#e8e8e8;font-size:0.85rem;'>{network_names}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:0.9rem;'>📡 Network</div><div style='color:#e8e8e8;font-size:0.85rem;'>N/A</div>", unsafe_allow_html=True)
    
    with col2:
        genres = series_data.get('genres', [])
        if genres:
            genre_names = ', '.join([g.get('name', '') for g in genres[:3]])
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:0.9rem;'>🎭 Genres</div><div style='color:#e8e8e8;font-size:0.85rem;'>{genre_names}</div>", unsafe_allow_html=True)
    
    with col3:
        status = series_data.get('status', 'Unknown')
        st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:0.9rem;'>🎬 Status</div><div style='color:#e8e8e8;font-size:0.85rem;'>{status}</div>", unsafe_allow_html=True)
    
    # Production companies
    companies = series_data.get('production_companies', [])
    if companies:
        st.divider()
        st.markdown("<strong>🏢 Production Companies:</strong>")
        company_names = ', '.join([c.get('name', '') for c in companies[:3]])
        st.write(company_names)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'mood_filter' not in st.session_state:
    st.session_state.mood_filter = None

# Header
st.markdown("""
<div class="header">
    <h1>🎬 CineMatch</h1>
    <p>✨ AI-Powered Recommendations • 150K+ Titles • Movies & Series ✨</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📌 Navigation")
    page = st.radio("", ["🏠 Home", "🎬 Movies", "📺 TV Series", "🇮🇳 Indian", "📊 Analytics"], label_visibility="collapsed")
    st.session_state.page = page
    
    st.divider()
    st.markdown("### 📊 Database")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{len(movies_df):,}</div><div class='stat-label'>Titles</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{movies_df['language'].nunique()}</div><div class='stat-label'>Languages</div></div>", unsafe_allow_html=True)

# HOME
if st.session_state.page == "🏠 Home":
    st.markdown("## 🎯 What's Your Mood Today?")
    
    moods = [
        ("😊 Happy", ["Comedy", "Animation"]),
        ("😢 Sad", ["Drama"]),
        ("🎢 Thrilled", ["Action", "Thriller"]),
        ("🤔 Thoughtful", ["Mystery", "Drama"]),
        ("💪 Motivated", ["Biography", "Sport"]),
        ("😌 Relaxed", ["Romance"]),
        ("🤩 Excited", ["Adventure", "Fantasy"]),
        ("🌟 Inspired", ["Documentary"]),
    ]
    
    cols = st.columns(4)
    for i, (mood_label, genres) in enumerate(moods):
        with cols[i % 4]:
            if st.button(mood_label, use_container_width=True, key=f"mood_{i}"):
                st.session_state.mood_filter = genres
                st.rerun()
    
    st.divider()
    st.markdown("## 📊 Quick Stats")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{movies_df['averageRating'].mean():.1f}</div><div class='stat-label'>Avg Rating</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{interactions_df['user_id'].nunique():,}</div><div class='stat-label'>Users</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{int(movies_df['startYear'].max())}</div><div class='stat-label'>Latest</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{len(set(','.join(movies_df['genres'].dropna()).split(',')))}</div><div class='stat-label'>Genres</div></div>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("## 🎬 Search Movie Collections")
    search_query = st.text_input("Search for movie franchises or series collections...", placeholder="e.g., Marvel, Star Wars, Fast and Furious")
    
    if search_query:
        collections = search_tmdb_collections(search_query)
        if collections:
            st.success(f"Found {len(collections)} collections")
            
            for collection in collections[:5]:
                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_path = collection.get('poster_path')
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                        img = get_image(poster_url)
                        if img:
                            st.image(img, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {collection.get('name', 'Unknown')}")
                    st.write(collection.get('overview', 'No description available')[:200] + "...")
                    
                    if st.button(f"View Collection", key=f"collection_{collection['id']}"):
                        collection_details = get_tmdb_collection_details(collection['id'])
                        if collection_details:
                            st.markdown(f"### {collection_details.get('name', 'Unknown')}")
                            st.write(collection_details.get('overview', ''))
                            
                            parts = collection_details.get('parts', [])
                            if parts:
                                st.markdown(f"**📽️ {len(parts)} Movies in this collection:**")
                                for part in parts[:10]:
                                    st.write(f"- {part.get('title', 'Unknown')} ({part.get('release_date', 'N/A')[:4]})")
    
    st.divider()
    
    st.markdown("## 📺 Search TV Series")
    tv_search_query = st.text_input("Search for TV series...", placeholder="e.g., Breaking Bad, Game of Thrones, The Office")
    
    if tv_search_query:
        # Search TV series from TMDB
        try:
            bearer_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MTVhZTQ0MmJkMmY2YWFjNDhjM2M4ODc5MjViNGQ1ZiIsIm5iZiI6MTc2ODEwMDE0My45MDIwMDAyLCJzdWIiOiI2OTYzMTEyZjJmNDU2YzIwNGExNzFlMDYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.k1Astjr0GL-Waej7WNvU_gbIhBF6w6FgHvs-4dexCXE"
            url = "https://api.themoviedb.org/3/search/tv"
            headers = {"accept": "application/json", "Authorization": f"Bearer {bearer_token}"}
            params = {"query": tv_search_query, "language": "en-US", "page": 1}
            
            response = requests.get(url, headers=headers, params=params, timeout=8)
            if response.status_code == 200:
                tv_results = response.json().get('results', [])
                if tv_results:
                    st.success(f"Found {len(tv_results)} TV series")
                    for series in tv_results[:5]:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if series.get('poster_path'):
                                poster_url = f"https://image.tmdb.org/t/p/w200{series['poster_path']}"
                                img = get_image(poster_url)
                                if img:
                                    st.image(img, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"### {series.get('name', 'Unknown')}")
                            st.write(series.get('overview', 'No description')[:150] + "...")
                            st.write(f"⭐ {series.get('vote_average', 0):.1f}/10 • {series.get('first_air_date', 'N/A')}")
                            
                            if st.button(f"View Full Details", key=f"tv_series_{series['id']}"):
                                series_details = get_tmdb_tv_series_details(series['id'])
                                if series_details:
                                    show_tv_series_detail(series_details)
        except:
            st.error("Could not search TV series")
    
    st.divider()
    
    if st.session_state.mood_filter:
        st.markdown(f"## 🎬 {st.session_state.mood_filter[0]} Recommendations")
        filtered = movies_df.copy()
        for g in st.session_state.mood_filter:
            filtered = filtered[filtered['genres'].str.contains(g, na=False)]
        filtered = filtered.sort_values('averageRating', ascending=False).head(20)
        st.success(f"Found {len(filtered)} titles")
        
        for _, movie in filtered.iterrows():
            show_movie_full_detail(movie)
    else:
        st.markdown("## 🔥 Trending Today (Real-time)")
        daily_trending = get_tmdb_daily_trending()
        
        if daily_trending:
            st.info("📊 Real-time trending from TMDB - Updated every 30 minutes")
            # Show first 5 trending items from local database
            trending = movies_df.nlargest(5, 'numVotes')
            for _, movie in trending.iterrows():
                show_movie_full_detail(movie)
        else:
            st.markdown("## 🌟 Trending Movies")
            trending = movies_df.nlargest(5, 'numVotes')
            for _, movie in trending.iterrows():
                show_movie_full_detail(movie)
        
        st.divider()
        st.markdown("## 📺 Trending TV Series")
        tv_trending = movies_df[movies_df['titleType'].isin(['tvSeries', 'tvMovie'])].nlargest(5, 'numVotes')
        
        for _, tv_series in tv_trending.iterrows():
            show_movie_full_detail(tv_series)

# TOP RATED
# MOVIES PAGE
elif st.session_state.page == "🎬 Movies":
    st.markdown("## 🎬 Movies")
    m_tabs = st.tabs(["🌟 Top Rated", "🎭 By Genre", "🌍 By Language"])
    
    # Filter for movies only
    movies_only = movies_df[movies_df['titleType'].isin(['movie', 'tvMovie'])]

    with m_tabs[0]:
        st.markdown("### 🌟 Top Rated Movies")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_rating, max_rating = st.slider("Rating Range", 1.0, 10.0, (7.0, 10.0), step=0.1, key="m_rate")
        with c2:
            year_min, year_max = st.slider("Year", 1960, 2024, (2010, 2024), key="m_year")
        with c3:
            langs = st.multiselect("Languages", sorted(movies_only['language'].dropna().unique()), default=['English', 'Hindi'], max_selections=5, key="m_lang")
        
        filtered = movies_only[
            (movies_only['averageRating'] >= min_rating) &
            (movies_only['averageRating'] <= max_rating) &
            (movies_only['startYear'] >= year_min) &
            (movies_only['startYear'] <= year_max) &
            (movies_only['language'].isin(langs))
        ].sort_values('averageRating', ascending=False).head(20)
        
        st.success(f"Found {len(filtered)} titles")
        for _, movie in filtered.iterrows():
            show_movie_full_detail(movie)
            
    with m_tabs[1]:
        st.markdown("### 🎭 Movies by Genre")
        all_genres = sorted(set(','.join(movies_only['genres'].dropna()).split(',')))
        genre = st.selectbox("Select Genre", all_genres, key="m_genre")
        
        filtered = movies_only[movies_only['genres'].str.contains(genre, na=False)].sort_values('averageRating', ascending=False).head(20)
        st.success(f"Found {len(filtered)} titles in {genre}")
        for _, movie in filtered.iterrows():
            show_movie_full_detail(movie)
            
    with m_tabs[2]:
        st.markdown("### 🌍 Movies by Language")
        language = st.selectbox("Language", sorted(movies_only['language'].unique()), key="m_lang_sel")
        filtered = movies_only[movies_only['language'] == language].sort_values('averageRating', ascending=False).head(20)
        st.success(f"Found {len(filtered)} titles in {language}")
        for _, movie in filtered.iterrows():
            show_movie_full_detail(movie)

# TV SERIES PAGE
elif st.session_state.page == "📺 TV Series":
    st.markdown("## 📺 TV Series")
    tv_tabs = st.tabs(["🌟 Top Rated", "🎭 By Genre", "🌍 By Language"])
    
    # Filter for TV Series only
    tv_only = movies_df[movies_df['titleType'].isin(['tvSeries', 'tvMiniSeries'])]
    
    with tv_tabs[0]:
        st.markdown("### 🌟 Top Rated TV Series")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_rating, max_rating = st.slider("Rating Range", 1.0, 10.0, (7.5, 10.0), step=0.1, key="tv_rate")
        with c2:
            year_min, year_max = st.slider("Year", 1960, 2024, (2010, 2024), key="tv_year")
        with c3:
            langs = st.multiselect("Languages", sorted(tv_only['language'].dropna().unique()), default=['English'], max_selections=5, key="tv_lang")
        
        filtered = tv_only[
            (tv_only['averageRating'] >= min_rating) &
            (tv_only['averageRating'] <= max_rating) &
            (tv_only['startYear'] >= year_min) &
            (tv_only['startYear'] <= year_max) &
            (tv_only['language'].isin(langs))
        ].sort_values('averageRating', ascending=False).head(20)
        
        st.success(f"Found {len(filtered)} titles")
        for _, series in filtered.iterrows():
            show_movie_full_detail(series) # Using movie detail view as fallback/standard, or update if distinct needed
            
    with tv_tabs[1]:
        st.markdown("### 🎭 TV Series by Genre")
        all_genres = sorted(set(','.join(tv_only['genres'].dropna()).split(',')))
        genre = st.selectbox("Select Genre", all_genres, key="tv_genre")
        
        filtered = tv_only[tv_only['genres'].str.contains(genre, na=False)].sort_values('averageRating', ascending=False).head(20)
        st.success(f"Found {len(filtered)} titles in {genre}")
        for _, series in filtered.iterrows():
            show_movie_full_detail(series)
            
    with tv_tabs[2]:
        st.markdown("### 🌍 TV Series by Language")
        language = st.selectbox("Language", sorted(tv_only['language'].unique()), key="tv_lang_sel")
        filtered = tv_only[tv_only['language'] == language].sort_values('averageRating', ascending=False).head(20)
        st.success(f"Found {len(filtered)} titles in {language}")
        for _, series in filtered.iterrows():
            show_movie_full_detail(series)

# INDIAN MOVIES
elif st.session_state.page == "🇮🇳 Indian":
    st.markdown("## 🇮🇳 Indian Cinema")
    
    c1, c2 = st.columns(2)
    with c1:
        ind_langs = st.multiselect("Languages", sorted(indian_movies_df['language'].unique()), default=['Hindi'], max_selections=5)
    with c2:
        min_rating, max_rating = st.slider("Rating Range", 1.0, 10.0, (6.0, 10.0), step=0.1)
    
    filtered = indian_movies_df[
        (indian_movies_df['language'].isin(ind_langs)) &
        (indian_movies_df['averageRating'] >= min_rating) &
        (indian_movies_df['averageRating'] <= max_rating)
    ].sort_values('averageRating', ascending=False).head(20)
    
    st.success(f"Found {len(filtered)} titles")
    
    for _, movie in filtered.iterrows():
        show_movie_full_detail(movie)

# ANALYTICS
elif st.session_state.page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    t1, t2, t3, t4 = st.tabs(["Languages 🌍", "Ratings ⭐", "Streaming 📺", "Types 🎬"])
    
    with t1:
        lang_data = movies_df['language'].value_counts().head(20)
        fig = px.bar(x=lang_data.index, y=lang_data.values, color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(template="plotly_dark", height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        fig = px.histogram(movies_df, x='averageRating', nbins=30, color_discrete_sequence=['#4ECDC4'])
        fig.update_layout(template="plotly_dark", height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with t3:
        stream = {'Netflix': streaming_df['netflix'].sum(), 'Prime': streaming_df['prime_video'].sum(), 'Disney+': streaming_df['disney_plus'].sum()}
        fig = px.pie(values=list(stream.values()), names=list(stream.keys()), color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFE66D'])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with t4:
        type_data = movies_df['titleType'].value_counts()
        fig = px.bar(x=type_data.index, y=type_data.values, color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(template="plotly_dark", height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
