"""
🎬 CineMatch - Movie & TV Recommender
Upgraded with:
  A) AI similarity-based recommendations (TF-IDF cosine similarity)
  B) Watchlist & Watched list with session persistence
  C) "More Like This" similar-movie rows + loading spinners
  D) Paginated, filterable search with debounce-friendly UX

Drop improvements.py alongside this file and install:
  pip install scikit-learn Pillow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO

# ── NEW: import all improvements ─────────────────────────────
from improvements import (
    check_secrets,
    build_similarity_index,
    render_ai_recommendations,
    render_watchlist_buttons,
    render_watchlist_sidebar,
    render_full_watchlist_page,
    render_tmdb_similar_row,
    render_search_bar,
    render_filter_row,
    search_movies_df,
    paginate,
    reset_page,
)

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 CineMatch", page_icon="🎬", layout="wide")

# ── Startup secrets check ─────────────────────────────────────
if not check_secrets():
    st.stop()

# ── Premium Dark Theme CSS ────────────────────────────────────
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
.header h1 { color:white; font-size:2.8rem; font-weight:800; text-shadow:0 3px 10px rgba(0,0,0,.3); }
.header p  { color:rgba(255,255,255,.95); margin-top:.5rem; font-size:1.1rem; }
.movie-detail-card {
  background: linear-gradient(180deg, #1a1f3a 0%, #0f1429 100%);
  border: 2px solid rgba(255,107,107,.3);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 30px rgba(0,0,0,.4);
  animation: slideUp .6s ease-out;
}
.movie-detail-card:hover { transform:translateY(-5px); box-shadow:0 12px 40px rgba(0,0,0,.6); transition:all .3s ease; }
.movie-title-large { color:#FF6B6B; font-size:2rem; font-weight:800; margin-bottom:1rem; }
.movie-rating-large { color:#FFE66D; font-size:1.3rem; font-weight:800; margin-bottom:1.5rem; }
.meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; margin-bottom:2rem; }
.meta-box { background:rgba(255,107,107,.1); padding:1rem; border-radius:10px; border-left:3px solid #FF6B6B; }
.meta-label { color:#FFE66D; font-weight:700; font-size:.8rem; }
.meta-value { color:#e8e8e8; font-size:.95rem; margin-top:.3rem; }
.description-text {
  color:#9ca3af; line-height:1.8; font-size:1rem; margin-bottom:2rem;
  background:rgba(255,107,107,.05); padding:1.5rem; border-radius:10px;
  border-left:3px solid #4ECDC4;
}
.stat-card { background:linear-gradient(135deg,rgba(255,107,107,.15),rgba(78,205,196,.1)); border:2px solid rgba(255,107,107,.3); padding:1.3rem; border-radius:12px; text-align:center; }
.stat-value { color:#FF6B6B; font-size:2rem; font-weight:800; }
.stat-label { color:#9ca3af; font-size:.85rem; margin-top:.4rem; }
@keyframes slideUp { from{transform:translateY(20px);opacity:0} to{transform:translateY(0);opacity:1} }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────
@st.cache_resource
def load_data():
    movies       = pd.read_csv("data/imdb_movies.csv")
    streaming    = pd.read_csv("data/streaming_platforms.csv")
    interactions = pd.read_csv("data/user_interactions.csv")
    try:
        indian_movies = pd.read_csv("data/indian_movies.csv")
    except Exception:
        indian_movies = movies[movies["language"].isin(["Hindi","Tamil","Telugu","Kannada","Malayalam"])]
    return movies, streaming, interactions, indian_movies

movies_df, streaming_df, interactions_df, indian_movies_df = load_data()

# Pre-warm the similarity index in the background so first search is fast
_ = build_similarity_index(movies_df)


# ── TMDB helper functions (unchanged from original) ───────────
def verify_tmdb_authentication():
    url = "https://api.themoviedb.org/3/authentication"
    headers = {"accept":"application/json","Authorization":f"Bearer {st.secrets['tmdb_token']}"}
    try:
        return requests.get(url, headers=headers).text
    except Exception as e:
        return f"Authentication error: {e}"


@st.cache_data(ttl=7200, show_spinner=False)
def get_movie_details(title, year, imdb_id=None, content_type="movie"):
    details = {"poster":None,"plot":None,"director":None,"actors":None,"runtime":None,"writer":None}
    tmdb_id = None
    try:
        bearer_token = st.secrets["tmdb_token"]
        endpoint = "tv" if content_type and content_type.lower() in ["tv","tvseries","tvmovie"] else "movie"
        url = f"https://api.themoviedb.org/3/search/{endpoint}"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        params  = {"query":title,"year":int(year) if endpoint=="movie" else None,
                   "first_air_date_year":int(year) if endpoint=="tv" else None,"language":"en-US"}
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            results = res.json().get("results",[])
            if results:
                tmdb_id = results[0]["id"]
                poster_path = results[0].get("poster_path")
                if poster_path:
                    details["poster"] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                details["plot"] = results[0].get("overview") or details["plot"]
    except Exception:
        pass

    if tmdb_id:
        try:
            bearer_token = st.secrets["tmdb_token"]
            headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
            credits_res = requests.get(
                f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/credits",
                headers=headers, timeout=5)
            if credits_res.status_code == 200:
                credits_data = credits_res.json()
                directors = [m["name"] for m in credits_data.get("crew",[]) if m["job"]=="Director"]
                if directors: details["director"] = ", ".join(directors[:2])
                actors = [m["name"] for m in credits_data.get("cast",[])]
                if actors: details["actors"] = ", ".join(actors[:4])
            prov_res = requests.get(
                f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/watch/providers",
                headers=headers, timeout=5)
            if prov_res.status_code == 200:
                us_prov = prov_res.json().get("results",{}).get("US",{}).get("flatrate",[])
                if us_prov: details["streaming"] = [p["provider_name"] for p in us_prov]
            # Store TMDB ID for "More Like This"
            details["tmdb_id"] = tmdb_id
        except Exception:
            pass

    if not details["poster"] or not details["director"]:
        omdb_data = get_omdb_data(title, year, imdb_id)
        if omdb_data:
            if not details["poster"]:   details["poster"]    = omdb_data.get("poster")
            if not details["director"]: details["director"]  = omdb_data.get("director")
            if not details["plot"] or len(details.get("plot") or "") < 20:
                details["plot"] = omdb_data.get("plot")
    return details


def get_omdb_data(title, year, imdb_id=None):
    try:
        api_key = st.secrets["omdb_key"]
        if imdb_id:
            url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        else:
            clean = title.replace(":","").replace("?","").replace('"',"").strip()
            url = f"http://www.omdbapi.com/?t={clean}&y={int(year)}&apikey={api_key}&type=movie"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("Response") == "True":
                return {k: data.get(v) if data.get(v) and data.get(v) != "N/A" else None
                        for k, v in [("plot","Plot"),("director","Director"),
                                     ("actors","Actors"),("runtime","Runtime"),("writer","Writer")]}
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_image(url):
    try:
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_tmdb_daily_trending():
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = "https://api.themoviedb.org/3/trending/all/day"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=headers, params={"language":"en-US"}, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results",[])
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_tmdb_collections(query):
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = "https://api.themoviedb.org/3/search/collection"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        params  = {"query":query,"include_adult":"false","language":"en-US","page":1}
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results",[])
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_collection_details(collection_id):
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = f"https://api.themoviedb.org/3/collection/{collection_id}"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=headers, params={"language":"en-US"}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_trending():
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = "https://api.themoviedb.org/3/discover/movie"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        params  = {"include_adult":"false","include_video":"false","language":"en-US","page":1,"sort_by":"popularity.desc"}
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results",[])
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_trending_tv():
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = "https://api.themoviedb.org/3/discover/tv"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        params  = {"include_adult":"false","include_null_first_air_dates":"false",
                   "language":"en-US","page":1,"sort_by":"popularity.desc"}
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results",[])
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_tv_series_details(series_id):
    try:
        bearer_token = st.secrets["tmdb_token"]
        url = f"https://api.themoviedb.org/3/tv/{series_id}"
        headers = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=headers, params={"language":"en-US"}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# ── Movie/TV display helpers ──────────────────────────────────
def show_movie_full_detail(movie):
    st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
    col_poster, col_info = st.columns([1.1, 1.9])

    with st.spinner("Loading movie details…"):
        details = get_movie_details(
            movie["primaryTitle"], movie["startYear"],
            movie.get("tconst"), movie.get("titleType")
        )

    with col_poster:
        st.write("")
        if details and details.get("poster"):
            img = get_image(details["poster"])
            if img:
                st.image(img, use_container_width=True, output_format="JPEG")
            else:
                st.markdown("<div style='background:linear-gradient(135deg,#FF6B6B,#4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>🎬</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:linear-gradient(135deg,#FF6B6B,#4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📽️</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(f'<div class="movie-title-large">{movie["primaryTitle"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="movie-rating-large">⭐ {movie["averageRating"]}/10 <span style="font-size:.9rem;color:#9ca3af;font-weight:400">({int(movie["numVotes"]):,} votes)</span></div>', unsafe_allow_html=True)

        # Metadata
        runtime = details.get("runtime", str(movie.get("runtimeMinutes","N/A")) + "min") if details else str(movie.get("runtimeMinutes","N/A")) + "min"
        st.markdown(f"""
        <div class="meta-grid">
          <div class="meta-box"><div class="meta-label">📅 Year</div><div class="meta-value">{movie['startYear']}</div></div>
          <div class="meta-box"><div class="meta-label">🎬 Type</div><div class="meta-value">{movie.get('titleType','Movie').replace('tv','TV ').title()}</div></div>
          <div class="meta-box"><div class="meta-label">🌍 Language</div><div class="meta-value">{movie['language']}</div></div>
          <div class="meta-box"><div class="meta-label">⏱ Runtime</div><div class="meta-value">{runtime}</div></div>
        </div>
        """, unsafe_allow_html=True)

        plot = (details.get("plot") if details else None) or f"A {movie['startYear']} {movie['language']} production. Rated {movie['averageRating']}/10."
        st.markdown("#### 📝 Plot Summary")
        if len(plot) > 280:
            st.markdown(f'<div class="description-text">{plot[:280]}…</div>', unsafe_allow_html=True)
            with st.expander("Read Full Plot"):
                st.write(plot)
        else:
            st.markdown(f'<div class="description-text">{plot}</div>', unsafe_allow_html=True)

        if details and details.get("streaming"):
            st.markdown("#### 📺 Where to Watch")
            badges = "".join([f'<span style="background:rgba(78,205,196,.2);color:#4ECDC4;padding:4px 10px;border-radius:4px;margin-right:8px;font-size:.9rem;border:1px solid rgba(78,205,196,.4)">{p}</span>'
                              for p in details["streaming"]])
            st.markdown(f'<div style="margin-bottom:1.5rem">{badges}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            director = (details.get("director","N/A") if details else "N/A")
            st.markdown(f"#### 👤 Director\n<div style='color:#e8e8e8;margin-bottom:1rem'>{director}</div>", unsafe_allow_html=True)
        with c2:
            genres = str(movie.get("genres","")).replace(",", ", ")
            st.markdown(f"#### 🎭 Genres\n<div style='color:#e8e8e8;margin-bottom:1rem'>{genres}</div>", unsafe_allow_html=True)
        if details and details.get("actors"):
            st.markdown(f"#### 👥 Cast\n<div style='color:#e8e8e8;margin-bottom:1rem'>{details['actors']}</div>", unsafe_allow_html=True)

        st.divider()
        # ── B) Watchlist buttons ─────────────────────────────
        render_watchlist_buttons(movie)

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    # ── C) More Like This (TMDB live) ────────────────────────
    tmdb_id = details.get("tmdb_id") if details else None
    content_type = movie.get("titleType", "movie")
    if tmdb_id:
        render_tmdb_similar_row(tmdb_id, content_type)
    st.divider()

    # ── A) AI local recommendations ──────────────────────────
    render_ai_recommendations(movie, movies_df)


def show_tv_series_detail(series_data):
    st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
    col_poster, col_info = st.columns([1.1, 1.9])

    with col_poster:
        st.write("")
        if series_data.get("poster_path"):
            poster_url = f"https://image.tmdb.org/t/p/w500{series_data['poster_path']}"
            img = get_image(poster_url)
            if img:
                st.image(img, use_container_width=True, output_format="JPEG")
            else:
                st.markdown("<div style='background:linear-gradient(135deg,#FF6B6B,#4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📺</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:linear-gradient(135deg,#FF6B6B,#4ECDC4);height:320px;display:flex;align-items:center;justify-content:center;color:white;font-size:3rem;border-radius:12px;'>📺</div>", unsafe_allow_html=True)

    with col_info:
        title          = series_data.get("name","Unknown")
        rating         = series_data.get("vote_average",0)
        vote_count     = series_data.get("vote_count",0)
        first_air_date = series_data.get("first_air_date","N/A")
        last_air_date  = series_data.get("last_air_date","N/A")
        seasons        = len(series_data.get("seasons",[]))
        episodes       = sum(len(s.get("episodes",[])) for s in series_data.get("seasons",[]))
        overview       = series_data.get("overview","No description available")
        status         = series_data.get("status","Unknown")

        st.markdown(f"""
        <div class="movie-title-large">{title}</div>
        <div class="movie-rating-large">⭐ {rating:.1f}/10 · {vote_count:,} votes</div>
        <div class="meta-grid">
          <div class="meta-box"><div class="meta-label">📅 First Aired</div><div class="meta-value">{first_air_date}</div></div>
          <div class="meta-box"><div class="meta-label">🏁 Last Aired</div><div class="meta-value">{last_air_date}</div></div>
          <div class="meta-box"><div class="meta-label">📺 Seasons</div><div class="meta-value">{seasons}</div></div>
          <div class="meta-box"><div class="meta-label">🎬 Episodes</div><div class="meta-value">{episodes}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="description-text"><strong>📖 Overview:</strong><br/>{overview}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            networks = series_data.get("networks",[])
            net_names = ", ".join([n.get("name","") for n in networks[:2]]) if networks else "N/A"
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:.9rem;'>📡 Network</div><div style='color:#e8e8e8;font-size:.85rem;'>{net_names}</div>", unsafe_allow_html=True)
        with col2:
            genres = ", ".join([g.get("name","") for g in series_data.get("genres",[])[:3]])
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:.9rem;'>🎭 Genres</div><div style='color:#e8e8e8;font-size:.85rem;'>{genres}</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='color:#4ECDC4;font-weight:600;font-size:.9rem;'>🎬 Status</div><div style='color:#e8e8e8;font-size:.85rem;'>{status}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── C) More Like This for TV ─────────────────────────────
    if series_data.get("id"):
        render_tmdb_similar_row(series_data["id"], "tv")


# ── Session state init ────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "mood_filter" not in st.session_state:
    st.session_state.mood_filter = None
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <h1>🎬 CineMatch</h1>
  <p>✨ AI-Powered Recommendations • 150K+ Titles • Movies & Series ✨</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "🎬 Movies", "📺 TV Series", "🇮🇳 Indian", "❤️ My List", "📊 Analytics"],
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.divider()
    st.markdown("### 📊 Database")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{len(movies_df):,}</div><div class='stat-label'>Titles</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{movies_df['language'].nunique()}</div><div class='stat-label'>Languages</div></div>", unsafe_allow_html=True)

    st.divider()
    # ── B) Watchlist sidebar ─────────────────────────────────
    st.markdown("### ❤️ My List")
    render_watchlist_sidebar()


# ═══════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════
if st.session_state.page == "🏠 Home":

    st.markdown("## 🎯 What's Your Mood Today?")
    moods = [
        ("😊 Happy",     ["Comedy","Animation"]),
        ("😢 Sad",       ["Drama"]),
        ("🎢 Thrilled",  ["Action","Thriller"]),
        ("🤔 Thoughtful",["Mystery","Drama"]),
        ("💪 Motivated", ["Biography","Sport"]),
        ("😌 Relaxed",   ["Romance"]),
        ("🤩 Excited",   ["Adventure","Fantasy"]),
        ("🌟 Inspired",  ["Documentary"]),
    ]
    cols = st.columns(4)
    for i, (mood_label, genres) in enumerate(moods):
        with cols[i % 4]:
            if st.button(mood_label, use_container_width=True, key=f"mood_{i}"):
                st.session_state.mood_filter = genres
                reset_page("mood")
                st.rerun()

    # Show mood results
    if st.session_state.mood_filter:
        genres = st.session_state.mood_filter
        st.divider()
        st.markdown(f"### 🎬 Top picks for your mood")
        mood_df = movies_df[movies_df["genres"].apply(
            lambda g: any(gen in str(g) for gen in genres) if pd.notna(g) else False
        )].sort_values("averageRating", ascending=False)

        page_slice = paginate(mood_df, "mood")
        for _, movie in page_slice.iterrows():
            with st.expander(f"🎬 {movie['primaryTitle']} ({int(movie['startYear'])}) — ⭐ {movie['averageRating']}"):
                show_movie_full_detail(movie)

        if st.button("✖ Clear mood filter"):
            st.session_state.mood_filter = None
            reset_page("mood")
            st.rerun()

    st.divider()
    st.markdown("## 📊 Quick Stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='stat-card'><div class='stat-value'>{movies_df['averageRating'].mean():.1f}</div><div class='stat-label'>Avg Rating</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='stat-card'><div class='stat-value'>{interactions_df['user_id'].nunique():,}</div><div class='stat-label'>Users</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='stat-card'><div class='stat-value'>{int(movies_df['startYear'].max())}</div><div class='stat-label'>Latest</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='stat-card'><div class='stat-value'>{len(set(','.join(movies_df['genres'].dropna()).split(',')))}</div><div class='stat-label'>Genres</div></div>", unsafe_allow_html=True)

    st.divider()
    # ── D) Collection search with spinner ────────────────────
    st.markdown("## 🎬 Search Movie Collections")
    search_query = st.text_input(
        "Search for movie franchises or series collections…",
        placeholder="e.g. Marvel, Star Wars, Fast and Furious",
        key="collection_search",
    )

    if search_query:
        with st.spinner("🔍 Searching collections…"):
            collections = search_tmdb_collections(search_query)

        if collections:
            st.success(f"Found {len(collections)} collection(s)")
            for collection in collections[:5]:
                col1, col2 = st.columns([1, 3])
                with col1:
                    backdrop = collection.get("backdrop_path") or collection.get("poster_path")
                    if backdrop:
                        st.image(f"https://image.tmdb.org/t/p/w300{backdrop}", use_container_width=True)
                with col2:
                    st.markdown(f"#### {collection.get('name','Unknown')}")
                    with st.expander("View collection details"):
                        with st.spinner("Loading collection…"):
                            details = get_tmdb_collection_details(collection["id"])
                        if details:
                            parts = details.get("parts",[])
                            st.write(details.get("overview",""))
                            if parts:
                                st.markdown(f"**{len(parts)} movies in this collection:**")
                                p_cols = st.columns(min(4, len(parts)))
                                for j, part in enumerate(parts[:4]):
                                    with p_cols[j]:
                                        pp = part.get("poster_path")
                                        if pp:
                                            st.image(f"https://image.tmdb.org/t/p/w185{pp}", use_container_width=True)
                                        st.caption(f"{part.get('title','?')} ({(part.get('release_date') or '—')[:4]})")
        else:
            st.info("No collections found. Try a different search term.")

    st.divider()
    # ── Trending ─────────────────────────────────────────────
    st.markdown("## 🔥 Trending Today")
    with st.spinner("Fetching trending content…"):
        trending = get_tmdb_daily_trending()

    if trending:
        t_cols = st.columns(min(5, len(trending)))
        for i, item in enumerate(trending[:5]):
            with t_cols[i]:
                pp = item.get("poster_path")
                if pp:
                    st.image(f"https://image.tmdb.org/t/p/w185{pp}", use_container_width=True)
                name  = item.get("title") or item.get("name","?")
                score = item.get("vote_average",0)
                st.markdown(f"<div style='color:#FF6B6B;font-size:.82rem;font-weight:600;'>{name}</div><div style='color:#FFE66D;font-size:.78rem;'>⭐ {score:.1f}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: MOVIES  (D – search + pagination)
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "🎬 Movies":
    st.markdown("## 🎬 Movies")

    # Search bar
    query = render_search_bar("movies")

    # Filter row
    genre_f, lang_f, min_r, year_r = render_filter_row(movies_df, "movies")

    # Reset page when filters change
    filter_sig = (query, genre_f, lang_f, min_r, year_r)
    if st.session_state.get("_movies_last_filter") != filter_sig:
        st.session_state["_movies_last_filter"] = filter_sig
        reset_page("movies")

    # Search / filter
    with st.spinner("Searching…"):
        results = search_movies_df(
            movies_df, query,
            genre_filter=genre_f,
            lang_filter=lang_f,
            year_range=year_r,
            min_rating=min_r,
            content_type="Movies",
        )

    if results.empty:
        st.warning("No titles found. Try adjusting your filters.")
    else:
        # Paginate
        page_slice = paginate(results, "movies")
        for _, movie in page_slice.iterrows():
            genres = str(movie.get("genres","")).replace(",",", ")
            with st.expander(f"🎬 {movie['primaryTitle']} ({int(movie['startYear'])}) — ⭐ {movie['averageRating']} | {genres}"):
                show_movie_full_detail(movie)


# ═══════════════════════════════════════════════════════════
# PAGE: TV SERIES
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "📺 TV Series":
    st.markdown("## 📺 TV Series")

    tab_trending, tab_search = st.tabs(["🔥 Trending", "🔍 Search"])

    with tab_trending:
        with st.spinner("Loading trending series…"):
            tv_list = get_tmdb_trending_tv()

        if tv_list:
            t_cols = st.columns(min(4, len(tv_list)))
            for i, item in enumerate(tv_list[:8]):
                with t_cols[i % 4]:
                    pp = item.get("poster_path")
                    if pp:
                        st.image(f"https://image.tmdb.org/t/p/w185{pp}", use_container_width=True)
                    name  = item.get("name","?")
                    score = item.get("vote_average",0)
                    st.markdown(f"<div style='color:#FF6B6B;font-size:.82rem;font-weight:600;'>{name}</div><div style='color:#FFE66D;font-size:.78rem;'>⭐ {score:.1f}</div>", unsafe_allow_html=True)

                    if st.button("Details", key=f"tv_det_{item.get('id',i)}", use_container_width=True):
                        with st.spinner(f"Loading {name}…"):
                            series_data = get_tmdb_tv_series_details(item["id"])
                        if series_data:
                            show_tv_series_detail(series_data)

    with tab_search:
        tv_query = render_search_bar("tv")
        if tv_query:
            with st.spinner("Searching TV series…"):
                tv_results = search_movies_df(movies_df, tv_query, content_type="TV Series")
            if tv_results.empty:
                st.info("No local TV results found. Try the TMDB trending tab.")
            else:
                page_slice = paginate(tv_results, "tv_search")
                for _, movie in page_slice.iterrows():
                    with st.expander(f"📺 {movie['primaryTitle']} ({int(movie['startYear'])}) — ⭐ {movie['averageRating']}"):
                        show_movie_full_detail(movie)


# ═══════════════════════════════════════════════════════════
# PAGE: INDIAN CINEMA
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "🇮🇳 Indian":
    st.markdown("## 🇮🇳 Indian Cinema")

    ind_query = render_search_bar("indian")
    ind_genre, ind_lang, ind_rating, ind_year = render_filter_row(indian_movies_df, "indian")

    ind_sig = (ind_query, ind_genre, ind_lang, ind_rating, ind_year)
    if st.session_state.get("_indian_last_filter") != ind_sig:
        st.session_state["_indian_last_filter"] = ind_sig
        reset_page("indian")

    with st.spinner("Filtering…"):
        ind_results = search_movies_df(
            indian_movies_df, ind_query,
            genre_filter=ind_genre, lang_filter=ind_lang,
            year_range=ind_year, min_rating=ind_rating,
        )

    if ind_results.empty:
        st.warning("No titles found.")
    else:
        page_slice = paginate(ind_results, "indian")
        for _, movie in page_slice.iterrows():
            with st.expander(f"🎬 {movie['primaryTitle']} ({int(movie['startYear'])}) — ⭐ {movie['averageRating']} | {movie['language']}"):
                show_movie_full_detail(movie)


# ═══════════════════════════════════════════════════════════
# PAGE: MY LIST  (B – full watchlist page)
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "❤️ My List":
    render_full_watchlist_page(movies_df)


# ═══════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Rating Distribution")
        fig = px.histogram(movies_df, x="averageRating", nbins=40,
                           color_discrete_sequence=["#FF6B6B"],
                           labels={"averageRating":"Rating"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e8e8e8")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top Languages")
        lang_counts = movies_df["language"].value_counts().head(10).reset_index()
        lang_counts.columns = ["Language","Count"]
        fig2 = px.bar(lang_counts, x="Count", y="Language", orientation="h",
                      color_discrete_sequence=["#4ECDC4"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e8e8e8")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Genres Breakdown")
    all_genres = ",".join(movies_df["genres"].dropna()).split(",")
    genre_series = pd.Series([g.strip() for g in all_genres]).value_counts().head(15).reset_index()
    genre_series.columns = ["Genre","Count"]
    fig3 = px.bar(genre_series, x="Genre", y="Count", color="Count",
                  color_continuous_scale="Viridis")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e8e8e8")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Ratings Over Time")
    year_ratings = movies_df.groupby("startYear")["averageRating"].mean().reset_index()
    year_ratings = year_ratings[year_ratings["startYear"] >= 1980]
    fig4 = px.line(year_ratings, x="startYear", y="averageRating",
                   color_discrete_sequence=["#FFE66D"],
                   labels={"startYear":"Year","averageRating":"Avg Rating"})
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e8e8e8")
    st.plotly_chart(fig4, use_container_width=True)
