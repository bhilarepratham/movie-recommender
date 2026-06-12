"""
CineMatch Improvements Module
===============================
Drop this file alongside streamlit_app.py.

Provides:
  A) AI similarity-based recommender  (build_similarity_index, get_similar_movies_local)
  B) Watchlist / Favorites             (watchlist_add, watchlist_remove, render_watchlist_sidebar)
  C) Similar Movies via TMDB + spinner helpers
  D) Paginated, debounce-ready search  (paginated_results, search_movies_df)

Usage in streamlit_app.py:
    from improvements import *
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ─────────────────────────────────────────────────────────────
# A) AI SIMILARITY-BASED RECOMMENDER
# ─────────────────────────────────────────────────────────────

def _build_feature_text(row: pd.Series) -> str:
    """Combine genres, mood, language, and year bucket into one text blob."""
    genres = str(row.get("genres", "")) if pd.notna(row.get("genres")) else ""
    mood   = str(row.get("mood",   "")) if pd.notna(row.get("mood"))   else ""
    lang   = str(row.get("language", "")) if pd.notna(row.get("language")) else ""
    # Bucket year into decade so "2010s" has semantic weight
    try:
        decade = f"decade{int(row['startYear']) // 10 * 10}"
    except Exception:
        decade = ""
    # Repeat genres 3× so they dominate over metadata
    return f"{genres} {genres} {genres} {mood} {lang} {decade}".strip()


@st.cache_resource(show_spinner=False)
def build_similarity_index(movies_df: pd.DataFrame):
    """
    Build a TF-IDF cosine-similarity index over the movie catalogue.
    Returns (tfidf_matrix, vectorizer, index_series).
    Called once; cached for the session lifetime.
    """
    feature_texts = movies_df.apply(_build_feature_text, axis=1).fillna("")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(feature_texts)
    # Store the integer positional index aligned with movies_df
    index_series = pd.Series(range(len(movies_df)), index=movies_df.index)
    return tfidf_matrix, vectorizer, index_series


def get_similar_movies_local(
    movie_row: pd.Series,
    movies_df: pd.DataFrame,
    tfidf_matrix,
    index_series: pd.Series,
    n: int = 8,
    min_rating: float = 6.0,
) -> pd.DataFrame:
    """
    Return the top-n most similar movies to movie_row using cosine similarity.
    Excludes the source movie itself and titles below min_rating.
    """
    src_pos = index_series.get(movie_row.name)
    if src_pos is None:
        return pd.DataFrame()

    src_vec = tfidf_matrix[src_pos]
    cos_scores = cosine_similarity(src_vec, tfidf_matrix).flatten()

    # Zero out the source movie
    cos_scores[src_pos] = 0.0

    # Apply rating filter as a soft mask
    ratings = movies_df["averageRating"].fillna(0).values
    cos_scores[ratings < min_rating] *= 0.1

    top_indices = cos_scores.argsort()[::-1][:n * 3]  # over-fetch, then trim
    results = movies_df.iloc[top_indices].copy()
    results["_similarity"] = cos_scores[top_indices]
    results = results[results["_similarity"] > 0.05]
    return results.head(n)


def render_ai_recommendations(
    seed_movie: pd.Series,
    movies_df: pd.DataFrame,
) -> None:
    """
    Render 'Because you watched X' AI recommendation row.
    Call after showing movie detail.
    """
    with st.spinner("🤖 Finding similar titles…"):
        tfidf_matrix, vectorizer, index_series = build_similarity_index(movies_df)
        similar = get_similar_movies_local(seed_movie, movies_df, tfidf_matrix, index_series)

    if similar.empty:
        return

    title = seed_movie.get("primaryTitle", "this movie")
    st.markdown(f"### 🎯 Because you viewed **{title}**")
    cols = st.columns(min(4, len(similar)))
    for i, (_, row) in enumerate(similar.iterrows()):
        if i >= 4:
            break
        with cols[i]:
            _render_mini_card(row)


def _render_mini_card(row: pd.Series) -> None:
    """Compact card used in recommendation rows."""
    genres_raw = row.get("genres", "")
    genres = str(genres_raw)[:28] if isinstance(genres_raw, str) else "—"
    rating = row.get("averageRating", "N/A")
    year   = int(row.get("startYear", 0)) if pd.notna(row.get("startYear")) else "—"
    title  = row.get("primaryTitle", "Unknown")

    sim_pct = int(row.get("_similarity", 0) * 100)
    sim_badge = f'<span style="color:#4ECDC4;font-size:0.75rem;">🔗 {sim_pct}% match</span>' if sim_pct else ""

    st.markdown(f"""
    <div style="
        background:linear-gradient(180deg,#1a1f3a,#0f1429);
        border:1px solid rgba(255,107,107,0.25);
        border-radius:12px;
        padding:1rem;
        height:100%;
        ">
      <div style="color:#FF6B6B;font-weight:700;font-size:0.9rem;margin-bottom:0.3rem;
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{title}">
        {title}
      </div>
      <div style="color:#FFE66D;font-size:0.8rem;">⭐ {rating} &nbsp;·&nbsp; {year}</div>
      <div style="color:#9ca3af;font-size:0.75rem;margin-top:0.2rem;">{genres}</div>
      <div style="margin-top:0.4rem;">{sim_badge}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# B) WATCHLIST / FAVOURITES
# ─────────────────────────────────────────────────────────────

def _init_watchlist() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}   # tconst → row dict
    if "watched" not in st.session_state:
        st.session_state.watched = {}


def watchlist_add(movie_row: pd.Series, list_key: str = "watchlist") -> None:
    _init_watchlist()
    key = str(movie_row.get("tconst", movie_row.get("primaryTitle", id(movie_row))))
    st.session_state[list_key][key] = movie_row.to_dict()


def watchlist_remove(movie_row: pd.Series, list_key: str = "watchlist") -> None:
    _init_watchlist()
    key = str(movie_row.get("tconst", movie_row.get("primaryTitle", id(movie_row))))
    st.session_state[list_key].pop(key, None)


def is_in_watchlist(movie_row: pd.Series, list_key: str = "watchlist") -> bool:
    _init_watchlist()
    key = str(movie_row.get("tconst", movie_row.get("primaryTitle", id(movie_row))))
    return key in st.session_state.get(list_key, {})


def render_watchlist_buttons(movie_row: pd.Series) -> None:
    """Render ➕/✅ Watchlist and 👁 Watched buttons inline."""
    _init_watchlist()
    col1, col2, col3 = st.columns([1, 1, 2])
    in_wl   = is_in_watchlist(movie_row, "watchlist")
    in_wtch = is_in_watchlist(movie_row, "watched")

    title = movie_row.get("primaryTitle", "movie")
    key_base = str(movie_row.get("tconst", title))

    with col1:
        if in_wl:
            if st.button("✅ Saved", key=f"wl_rem_{key_base}", use_container_width=True):
                watchlist_remove(movie_row, "watchlist")
                st.rerun()
        else:
            if st.button("➕ Watchlist", key=f"wl_add_{key_base}", use_container_width=True):
                watchlist_add(movie_row, "watchlist")
                st.toast(f"Added **{title}** to your watchlist!", icon="🎬")

    with col2:
        if in_wtch:
            if st.button("👁 Watched", key=f"wt_rem_{key_base}", use_container_width=True,
                         help="Mark as unwatched"):
                watchlist_remove(movie_row, "watched")
                st.rerun()
        else:
            if st.button("👁 Mark watched", key=f"wt_add_{key_base}", use_container_width=True):
                watchlist_add(movie_row, "watched")
                st.toast(f"Marked **{title}** as watched!", icon="✅")


def render_watchlist_sidebar() -> None:
    """Render watchlist summary + counts in the sidebar."""
    _init_watchlist()
    wl_count  = len(st.session_state.watchlist)
    wt_count  = len(st.session_state.watched)

    if wl_count == 0 and wt_count == 0:
        st.sidebar.markdown(
            "<div style='color:#9ca3af;font-size:0.82rem;'>No saved titles yet.</div>",
            unsafe_allow_html=True,
        )
        return

    st.sidebar.markdown(f"""
    <div style="display:flex;gap:0.5rem;margin-bottom:0.5rem;">
      <div style="flex:1;background:rgba(255,107,107,0.12);border:1px solid rgba(255,107,107,0.3);
                  border-radius:8px;padding:0.6rem;text-align:center;">
        <div style="color:#FF6B6B;font-size:1.3rem;font-weight:800;">{wl_count}</div>
        <div style="color:#9ca3af;font-size:0.72rem;">To Watch</div>
      </div>
      <div style="flex:1;background:rgba(78,205,196,0.12);border:1px solid rgba(78,205,196,0.3);
                  border-radius:8px;padding:0.6rem;text-align:center;">
        <div style="color:#4ECDC4;font-size:1.3rem;font-weight:800;">{wt_count}</div>
        <div style="color:#9ca3af;font-size:0.72rem;">Watched</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if wl_count > 0:
        with st.sidebar.expander(f"📋 My Watchlist ({wl_count})"):
            for key, data in list(st.session_state.watchlist.items()):
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    f"<span style='color:#e8e8e8;font-size:0.8rem;'>{data.get('primaryTitle','?')}"
                    f" <span style='color:#9ca3af'>({int(data.get('startYear',0))})</span></span>",
                    unsafe_allow_html=True,
                )
                if c2.button("✕", key=f"sid_rem_{key}"):
                    st.session_state.watchlist.pop(key, None)
                    st.rerun()


def render_full_watchlist_page(movies_df: pd.DataFrame) -> None:
    """Full-page watchlist view; call when page == '❤️ My List'."""
    _init_watchlist()
    st.markdown("## ❤️ My List")

    tab1, tab2 = st.tabs([
        f"📋 To Watch ({len(st.session_state.watchlist)})",
        f"✅ Watched ({len(st.session_state.watched)})",
    ])

    for tab, list_key, label in [
        (tab1, "watchlist", "watchlist"),
        (tab2, "watched",   "watched"),
    ]:
        with tab:
            items = st.session_state.get(list_key, {})
            if not items:
                st.info(f"Your {label} is empty. Browse movies and hit ➕ / 👁 to add them!")
                continue

            # Build a mini-DataFrame for display
            rows = list(items.values())
            df = pd.DataFrame(rows)

            # Rating sort
            col_sort, col_clear = st.columns([3, 1])
            sort_by = col_sort.selectbox(
                "Sort by", ["Rating ↓", "Year ↓", "Title A-Z"],
                key=f"sort_{list_key}"
            )
            if col_clear.button("🗑 Clear all", key=f"clear_{list_key}"):
                st.session_state[list_key] = {}
                st.rerun()

            if sort_by == "Rating ↓":
                df = df.sort_values("averageRating", ascending=False)
            elif sort_by == "Year ↓":
                df = df.sort_values("startYear", ascending=False)
            else:
                df = df.sort_values("primaryTitle")

            # Display in grid
            cols = st.columns(4)
            for i, (_, row) in enumerate(df.iterrows()):
                row_s = pd.Series(row)
                with cols[i % 4]:
                    _render_mini_card(row_s)
                    in_wl = is_in_watchlist(row_s, list_key)
                    key_b = str(row_s.get("tconst", row_s.get("primaryTitle", i)))
                    if st.button("Remove", key=f"rm_{list_key}_{key_b}", use_container_width=True):
                        watchlist_remove(row_s, list_key)
                        st.rerun()

            # AI "Watch Next" suggestion from watchlist
            if list_key == "watched" and len(df) >= 2:
                st.divider()
                st.markdown("### 🤖 You might like next…")
                # Pick highest-rated watched movie as seed
                seed_row = df.sort_values("averageRating", ascending=False).iloc[0]
                seed_series = pd.Series(seed_row)
                # Align index to movies_df
                match = movies_df[movies_df["primaryTitle"] == seed_series.get("primaryTitle")]
                if not match.empty:
                    render_ai_recommendations(match.iloc[0], movies_df)


# ─────────────────────────────────────────────────────────────
# C) SIMILAR MOVIES VIA TMDB (live API) + SPINNER HELPERS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_similar_movies(tmdb_id: int, content_type: str = "movie") -> list:
    """
    Fetch TMDB /movie/{id}/recommendations or /tv/{id}/recommendations.
    Returns list of result dicts (title, poster_path, vote_average, etc.)
    """
    try:
        bearer_token = st.secrets["tmdb_token"]
        endpoint = "tv" if content_type in ("tv", "tvseries", "tvmovie") else "movie"
        url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/recommendations"
        headers = {"accept": "application/json", "Authorization": f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=headers, params={"language": "en-US", "page": 1}, timeout=6)
        if resp.status_code == 200:
            return resp.json().get("results", [])[:8]
    except Exception:
        pass
    return []


def render_tmdb_similar_row(tmdb_id: int, content_type: str = "movie") -> None:
    """Render 'More like this' row using TMDB recommendations API."""
    with st.spinner("🎬 Loading similar titles…"):
        results = get_tmdb_similar_movies(tmdb_id, content_type)

    if not results:
        return

    st.markdown("### 🍿 More Like This")
    cols = st.columns(min(4, len(results)))
    for i, item in enumerate(results[:4]):
        title     = item.get("title") or item.get("name", "Unknown")
        rating    = item.get("vote_average", 0)
        year_raw  = (item.get("release_date") or item.get("first_air_date") or "")[:4]
        poster    = item.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w185{poster}" if poster else None
        overview  = item.get("overview", "")[:100]

        with cols[i]:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            st.markdown(f"""
            <div style="color:#FF6B6B;font-weight:700;font-size:0.85rem;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                 title="{title}">{title}</div>
            <div style="color:#FFE66D;font-size:0.78rem;">⭐ {rating:.1f} · {year_raw}</div>
            <div style="color:#9ca3af;font-size:0.72rem;margin-top:0.2rem;">{overview}…</div>
            """, unsafe_allow_html=True)


def spinner_section(label: str):
    """Context manager: wraps a block in st.spinner."""
    return st.spinner(label)


# ─────────────────────────────────────────────────────────────
# D) PAGINATED SEARCH
# ─────────────────────────────────────────────────────────────

PAGE_SIZE = 12   # results per page


def search_movies_df(
    df: pd.DataFrame,
    query: str,
    genre_filter: str = "All",
    lang_filter:  str = "All",
    year_range: tuple = (1900, 2025),
    min_rating: float = 0.0,
    content_type: str = "All",
) -> pd.DataFrame:
    """
    Fast filtered search over the local movie DataFrame.
    Returns sorted results — caller handles pagination.
    """
    result = df.copy()

    if query and query.strip():
        q = query.strip().lower()
        mask = result["primaryTitle"].str.lower().str.contains(q, na=False)
        result = result[mask]

    if genre_filter != "All":
        result = result[result["genres"].str.contains(genre_filter, na=False)]

    if lang_filter != "All":
        result = result[result["language"] == lang_filter]

    if content_type != "All":
        type_map = {"Movies": "movie", "TV Series": "tvseries"}
        if content_type in type_map:
            result = result[result["titleType"] == type_map[content_type]]

    result = result[
        result["startYear"].between(year_range[0], year_range[1]) &
        (result["averageRating"] >= min_rating)
    ]

    return result.sort_values("averageRating", ascending=False)


def render_search_bar(key_prefix: str = "search") -> str:
    """
    Render a styled search bar. Returns the current query string.
    Uses a slight key trick to avoid full reruns on every keystroke.
    """
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.06) !important;
        border: 1.5px solid rgba(255,107,107,0.35) !important;
        border-radius: 10px !important;
        color: #e8e8e8 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 3px rgba(255,107,107,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    return st.text_input(
        "🔍 Search titles…",
        placeholder="e.g. Inception, The Dark Knight, Parasite",
        key=f"{key_prefix}_input",
        label_visibility="collapsed",
    )


def render_filter_row(
    df: pd.DataFrame,
    key_prefix: str = "filter",
) -> tuple:
    """
    Render genre / language / rating / year filters. Returns (genre, lang, rating, year_range).
    """
    all_genres = sorted({
        g.strip()
        for genres in df["genres"].dropna()
        for g in genres.split(",")
        if g.strip()
    })
    all_langs = ["All"] + sorted(df["language"].dropna().unique().tolist())

    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])

    with col1:
        genre = st.selectbox(
            "Genre", ["All"] + all_genres,
            key=f"{key_prefix}_genre"
        )
    with col2:
        lang = st.selectbox(
            "Language", all_langs,
            key=f"{key_prefix}_lang"
        )
    with col3:
        min_rating = st.number_input(
            "Min ⭐", min_value=0.0, max_value=10.0,
            value=6.0, step=0.5, format="%.1f",
            key=f"{key_prefix}_rating"
        )
    with col4:
        current_year = int(df["startYear"].max()) if not df.empty else 2025
        year_range = st.slider(
            "Year range",
            min_value=1900, max_value=current_year,
            value=(1990, current_year),
            key=f"{key_prefix}_year"
        )

    return genre, lang, min_rating, year_range


def paginate(df: pd.DataFrame, key_prefix: str = "page") -> pd.DataFrame:
    """
    Given a full result DataFrame, render pagination controls
    and return only the current-page slice.
    """
    total = len(df)
    if total == 0:
        return df

    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    if f"{key_prefix}_num" not in st.session_state:
        st.session_state[f"{key_prefix}_num"] = 1

    # Clamp page number
    page_num = max(1, min(st.session_state[f"{key_prefix}_num"], total_pages))

    start = (page_num - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE
    page_slice = df.iloc[start:end]

    # Pagination controls
    col_info, col_prev, col_jump, col_next = st.columns([3, 1, 1, 1])
    with col_info:
        st.markdown(
            f"<span style='color:#9ca3af;font-size:0.85rem;'>"
            f"Showing {start+1}–{min(end, total)} of <b style='color:#FF6B6B'>{total:,}</b> results"
            f"</span>",
            unsafe_allow_html=True,
        )
    with col_prev:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=(page_num <= 1)):
            st.session_state[f"{key_prefix}_num"] = page_num - 1
            st.rerun()
    with col_jump:
        st.markdown(
            f"<div style='text-align:center;color:#e8e8e8;padding-top:0.4rem'>"
            f"Page {page_num}/{total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=(page_num >= total_pages)):
            st.session_state[f"{key_prefix}_num"] = page_num + 1
            st.rerun()

    return page_slice


def reset_page(key_prefix: str = "page") -> None:
    """Call when search query / filters change to jump back to page 1."""
    st.session_state[f"{key_prefix}_num"] = 1


# ─────────────────────────────────────────────────────────────
# SECRETS VALIDATION (startup guard)
# ─────────────────────────────────────────────────────────────

def check_secrets() -> bool:
    """Return True if required secrets exist; render an error banner if not."""
    missing = []
    for key in ("tmdb_token", "omdb_key"):
        try:
            _ = st.secrets[key]
        except (KeyError, FileNotFoundError):
            missing.append(key)

    if missing:
        st.error(
            f"⚠️ **Missing API secrets:** `{', '.join(missing)}`\n\n"
            "Add them to `.streamlit/secrets.toml`:\n"
            "```toml\ntmdb_token = \"Bearer eyJ...\"\nomdb_key = \"abc123\"\n```",
            icon="🔑",
        )
        return False
    return True
