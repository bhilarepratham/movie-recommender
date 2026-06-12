"""
CineMatch Improvements Module  (v2 — Hybrid Recommender)
=========================================================
Drop alongside streamlit_app.py.

A) Hybrid Recommender  — fixes repetition with 4 strategies:
     1. Content similarity  (TF-IDF on genres + mood + language + decade)
     2. Collaborative-style signals  (popularity-weighted genre overlap)
     3. Watch-history exclusion  (never re-surface already-seen titles)
     4. Diversity injection  (MMR — Maximal Marginal Relevance)
   Result: varied, non-repetitive, personalised recommendations.

B) Watchlist / Favourites
C) TMDB "More Like This" row + spinner helpers
D) Paginated, filterable search
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# A) HYBRID RECOMMENDER
# ─────────────────────────────────────────────────────────────

# ── 1. Feature engineering ───────────────────────────────────

def _build_feature_text(row: pd.Series) -> str:
    """
    Rich text blob per movie for TF-IDF.
    Genres are repeated 3× (strongest signal).
    Mood 2×, language 1×, decade 1×, vote-tier 1×.
    """
    genres = str(row.get("genres", "")) if pd.notna(row.get("genres")) else ""
    mood   = str(row.get("mood",   "")) if pd.notna(row.get("mood"))   else ""
    lang   = str(row.get("language","")) if pd.notna(row.get("language")) else ""

    try:
        decade = f"decade{int(row['startYear']) // 10 * 10}"
    except Exception:
        decade = ""

    # Vote tier: blockbuster / popular / niche — adds diversity axis
    try:
        votes = int(row.get("numVotes", 0))
        if votes > 500_000:
            tier = "blockbuster blockbuster"
        elif votes > 50_000:
            tier = "popular"
        else:
            tier = "niche"
    except Exception:
        tier = ""

    # Normalise genre separators so "Action,Adventure" → "Action Adventure"
    genres_clean = genres.replace(",", " ").replace("|", " ")
    mood_clean   = mood.replace(",", " ")

    return (
        f"{genres_clean} {genres_clean} {genres_clean} "
        f"{mood_clean} {mood_clean} "
        f"{lang} {decade} {tier}"
    ).strip()


# ── 2. Index (cached for full session) ───────────────────────

@st.cache_resource(show_spinner=False)
def build_similarity_index(movies_df: pd.DataFrame):
    """
    Build TF-IDF matrix + positional index.
    Returns (tfidf_matrix, index_series).
    """
    texts = movies_df.apply(_build_feature_text, axis=1).fillna("")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12_000,
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vec.fit_transform(texts)
    index  = pd.Series(range(len(movies_df)), index=movies_df.index)
    return matrix, index


# ── 3. MMR diversity re-ranker ────────────────────────────────

def _mmr_rerank(
    candidate_indices: np.ndarray,
    cos_scores: np.ndarray,
    tfidf_matrix,
    n: int,
    lambda_: float = 0.65,
) -> list:
    """
    Maximal Marginal Relevance:
    Iteratively pick the candidate that maximises:
        λ · relevance  −  (1-λ) · max_similarity_to_already_picked

    lambda_=1.0 → pure relevance (no diversity)
    lambda_=0.0 → pure diversity
    0.65 gives a good relevance-diversity balance.
    """
    selected   = []
    remaining  = list(candidate_indices)

    while remaining and len(selected) < n:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: cos_scores[i])
        else:
            sel_matrix = tfidf_matrix[selected]
            best, best_score = None, -np.inf
            for i in remaining:
                rel  = cos_scores[i]
                # Max similarity to any already-selected item
                sim_to_sel = cosine_similarity(
                    tfidf_matrix[i], sel_matrix
                ).max()
                score = lambda_ * rel - (1 - lambda_) * sim_to_sel
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


# ── 4. Collaborative popularity signal ───────────────────────

def _popularity_boost(movies_df: pd.DataFrame) -> np.ndarray:
    """
    Soft popularity weight ∈ [0, 1].
    Uses log-scaled vote counts so blockbusters don't dominate.
    """
    votes  = movies_df["numVotes"].fillna(1).clip(lower=1).values.astype(float)
    log_v  = np.log1p(votes)
    normed = (log_v - log_v.min()) / (log_v.max() - log_v.min() + 1e-9)
    return normed   # shape: (n_movies,)


# ── 5. Main recommendation function ──────────────────────────

def get_hybrid_recommendations(
    seed_movie: pd.Series,
    movies_df: pd.DataFrame,
    tfidf_matrix,
    index_series: pd.Series,
    n: int = 8,
    min_rating: float = 5.5,
    diversity_lambda: float = 0.65,
    exclude_titles: set = None,   # already-watched titles
) -> pd.DataFrame:
    """
    Hybrid recommender combining:
      • Content similarity  (TF-IDF cosine)
      • Popularity signal   (log vote boost, weight 0.15)
      • Watch-history exclusion
      • MMR diversity re-ranking

    Returns top-n DataFrame with _score and _similarity columns.
    """
    src_pos = index_series.get(seed_movie.name)
    if src_pos is None:
        return pd.DataFrame()

    # ── Content similarity ───────────────────────────────────
    src_vec     = tfidf_matrix[src_pos]
    cos_scores  = cosine_similarity(src_vec, tfidf_matrix).flatten()
    cos_scores[src_pos] = 0.0          # exclude self

    # ── Popularity boost (15 % weight) ───────────────────────
    pop_boost   = _popularity_boost(movies_df)
    hybrid      = 0.85 * cos_scores + 0.15 * pop_boost

    # ── Rating filter ─────────────────────────────────────────
    ratings = movies_df["averageRating"].fillna(0).values
    hybrid[ratings < min_rating] *= 0.05

    # ── Watch-history exclusion ───────────────────────────────
    if exclude_titles:
        for idx, row in movies_df.iterrows():
            if row.get("primaryTitle") in exclude_titles:
                pos = index_series.get(idx)
                if pos is not None:
                    hybrid[pos] = 0.0

    # ── Pre-filter: top 40 by hybrid score for MMR ───────────
    top40_idx   = hybrid.argsort()[::-1][:40]
    top40_idx   = top40_idx[hybrid[top40_idx] > 0.05]   # relevance floor

    if len(top40_idx) == 0:
        return pd.DataFrame()

    # ── MMR diversity re-rank ─────────────────────────────────
    diverse_idx = _mmr_rerank(
        candidate_indices=top40_idx,
        cos_scores=hybrid,
        tfidf_matrix=tfidf_matrix,
        n=n,
        lambda_=diversity_lambda,
    )

    result = movies_df.iloc[diverse_idx].copy()
    result["_similarity"] = cos_scores[diverse_idx]
    result["_score"]      = hybrid[diverse_idx]
    return result.reset_index(drop=True)


# ── 6. UI rendering ──────────────────────────────────────────

def _render_mini_card(row: pd.Series) -> None:
    genres_raw = row.get("genres", "")
    genres     = str(genres_raw)[:30] if isinstance(genres_raw, str) else "—"
    rating     = row.get("averageRating", "N/A")
    year       = int(row["startYear"]) if pd.notna(row.get("startYear")) else "—"
    title      = row.get("primaryTitle", "Unknown")
    sim_pct    = int(row.get("_similarity", 0) * 100)
    score_pct  = int(row.get("_score", 0) * 100)

    match_badge = (
        f'<span style="color:#4ECDC4;font-size:.72rem;">🔗 {sim_pct}% match</span>'
        if sim_pct else ""
    )

    st.markdown(f"""
    <div style="
        background:linear-gradient(180deg,#1a1f3a,#0f1429);
        border:1px solid rgba(255,107,107,0.25);
        border-radius:12px;padding:0.9rem;height:100%;
        transition:border-color .2s;">
      <div style="color:#FF6B6B;font-weight:700;font-size:.88rem;margin-bottom:.3rem;
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
           title="{title}">{title}</div>
      <div style="color:#FFE66D;font-size:.78rem;">⭐ {rating} &nbsp;·&nbsp; {year}</div>
      <div style="color:#9ca3af;font-size:.72rem;margin-top:.2rem;
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{genres}</div>
      <div style="margin-top:.4rem;">{match_badge}</div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_recommendations(
    seed_movie: pd.Series,
    movies_df: pd.DataFrame,
    label: str = "Because you viewed",
    n: int = 8,
    diversity: float = 0.65,
) -> None:
    """
    Render the full 'Because you viewed X' recommendation row.
    Respects the user's watched list to avoid repetition.
    """
    watched_titles = {
        d.get("primaryTitle")
        for d in st.session_state.get("watched", {}).values()
    }

    with st.spinner("🤖 Finding diverse recommendations…"):
        matrix, index = build_similarity_index(movies_df)
        recs = get_hybrid_recommendations(
            seed_movie, movies_df, matrix, index,
            n=n,
            diversity_lambda=diversity,
            exclude_titles=watched_titles,
        )

    if recs.empty:
        return

    title = seed_movie.get("primaryTitle", "this movie")
    st.markdown(f"### 🎯 {label} **{title}**")

    # ── Diversity slider (user-tunable) ──────────────────────
    with st.expander("⚙️ Tune recommendations", expanded=False):
        new_div = st.slider(
            "Diversity  ←  More varied  |  More similar  →",
            min_value=0.3, max_value=1.0, value=diversity,
            step=0.05,
            key=f"div_slider_{seed_movie.get('tconst','x')}",
        )
        new_n = st.slider(
            "Number of recommendations",
            min_value=4, max_value=16, value=n, step=2,
            key=f"n_slider_{seed_movie.get('tconst','x')}",
        )
        if new_div != diversity or new_n != n:
            with st.spinner("Recalculating…"):
                recs = get_hybrid_recommendations(
                    seed_movie, movies_df, matrix, index,
                    n=new_n,
                    diversity_lambda=new_div,
                    exclude_titles=watched_titles,
                )

    n_cols = min(4, len(recs))
    cols   = st.columns(n_cols)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % n_cols]:
            _render_mini_card(row)


# ─────────────────────────────────────────────────────────────
# B) WATCHLIST / FAVOURITES
# ─────────────────────────────────────────────────────────────

def _init_watchlist() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}
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
    _init_watchlist()
    in_wl   = is_in_watchlist(movie_row, "watchlist")
    in_wtch = is_in_watchlist(movie_row, "watched")
    title   = movie_row.get("primaryTitle", "movie")
    key_b   = str(movie_row.get("tconst", title))

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if in_wl:
            if st.button("✅ Saved", key=f"wl_rem_{key_b}", use_container_width=True):
                watchlist_remove(movie_row, "watchlist")
                st.rerun()
        else:
            if st.button("➕ Watchlist", key=f"wl_add_{key_b}", use_container_width=True):
                watchlist_add(movie_row, "watchlist")
                st.toast(f"Added **{title}** to your watchlist!", icon="🎬")
    with col2:
        if in_wtch:
            if st.button("👁 Watched", key=f"wt_rem_{key_b}", use_container_width=True,
                         help="Mark as unwatched"):
                watchlist_remove(movie_row, "watched")
                st.rerun()
        else:
            if st.button("👁 Mark watched", key=f"wt_add_{key_b}", use_container_width=True):
                watchlist_add(movie_row, "watched")
                st.toast(f"Marked **{title}** as watched!", icon="✅")


def render_watchlist_sidebar() -> None:
    _init_watchlist()
    wl_count = len(st.session_state.watchlist)
    wt_count = len(st.session_state.watched)

    if wl_count == 0 and wt_count == 0:
        st.sidebar.markdown(
            "<div style='color:#9ca3af;font-size:.82rem;'>No saved titles yet.</div>",
            unsafe_allow_html=True,
        )
        return

    st.sidebar.markdown(f"""
    <div style="display:flex;gap:.5rem;margin-bottom:.5rem;">
      <div style="flex:1;background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.3);
                  border-radius:8px;padding:.6rem;text-align:center;">
        <div style="color:#FF6B6B;font-size:1.3rem;font-weight:800;">{wl_count}</div>
        <div style="color:#9ca3af;font-size:.72rem;">To Watch</div>
      </div>
      <div style="flex:1;background:rgba(78,205,196,.12);border:1px solid rgba(78,205,196,.3);
                  border-radius:8px;padding:.6rem;text-align:center;">
        <div style="color:#4ECDC4;font-size:1.3rem;font-weight:800;">{wt_count}</div>
        <div style="color:#9ca3af;font-size:.72rem;">Watched</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if wl_count > 0:
        with st.sidebar.expander(f"📋 My Watchlist ({wl_count})"):
            for key, data in list(st.session_state.watchlist.items()):
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    f"<span style='color:#e8e8e8;font-size:.8rem;'>{data.get('primaryTitle','?')}"
                    f" <span style='color:#9ca3af'>({int(data.get('startYear',0))})</span></span>",
                    unsafe_allow_html=True,
                )
                if c2.button("✕", key=f"sid_rem_{key}"):
                    st.session_state.watchlist.pop(key, None)
                    st.rerun()


def render_full_watchlist_page(movies_df: pd.DataFrame) -> None:
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

            df = pd.DataFrame(list(items.values()))
            col_sort, col_clear = st.columns([3, 1])
            sort_by = col_sort.selectbox(
                "Sort by", ["Rating ↓","Year ↓","Title A-Z"],
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

            cols = st.columns(4)
            for i, (_, row) in enumerate(df.iterrows()):
                row_s = pd.Series(row)
                with cols[i % 4]:
                    _render_mini_card(row_s)
                    key_b = str(row_s.get("tconst", row_s.get("primaryTitle", i)))
                    if st.button("Remove", key=f"rm_{list_key}_{key_b}", use_container_width=True):
                        watchlist_remove(row_s, list_key)
                        st.rerun()

            # "Watch Next" powered by hybrid recommender
            if list_key == "watched" and len(df) >= 2:
                st.divider()
                st.markdown("### 🤖 Watch Next — based on your history")
                seed_row  = df.sort_values("averageRating", ascending=False).iloc[0]
                seed_s    = pd.Series(seed_row)
                match     = movies_df[movies_df["primaryTitle"] == seed_s.get("primaryTitle")]
                if not match.empty:
                    render_ai_recommendations(
                        match.iloc[0], movies_df,
                        label="Because you highly rated",
                        n=8, diversity=0.7,
                    )


# ─────────────────────────────────────────────────────────────
# C) TMDB "MORE LIKE THIS" + SPINNER HELPERS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_similar_movies(tmdb_id: int, content_type: str = "movie") -> list:
    try:
        bearer_token = st.secrets["tmdb_token"]
        endpoint = "tv" if content_type in ("tv","tvseries","tvmovie") else "movie"
        url  = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/recommendations"
        hdrs = {"accept":"application/json","Authorization":f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=hdrs, params={"language":"en-US","page":1}, timeout=6)
        if resp.status_code == 200:
            return resp.json().get("results",[])[:8]
    except Exception:
        pass
    return []


def render_tmdb_similar_row(tmdb_id: int, content_type: str = "movie") -> None:
    with st.spinner("🎬 Loading similar titles…"):
        results = get_tmdb_similar_movies(tmdb_id, content_type)
    if not results:
        return
    st.markdown("### 🍿 More Like This")
    cols = st.columns(min(4, len(results)))
    for i, item in enumerate(results[:4]):
        title    = item.get("title") or item.get("name","Unknown")
        rating   = item.get("vote_average", 0)
        year_raw = (item.get("release_date") or item.get("first_air_date") or "")[:4]
        poster   = item.get("poster_path")
        overview = item.get("overview","")[:100]
        with cols[i]:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w185{poster}", use_container_width=True)
            st.markdown(f"""
            <div style="color:#FF6B6B;font-weight:700;font-size:.85rem;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                 title="{title}">{title}</div>
            <div style="color:#FFE66D;font-size:.78rem;">⭐ {rating:.1f} · {year_raw}</div>
            <div style="color:#9ca3af;font-size:.72rem;margin-top:.2rem;">{overview}…</div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# D) PAGINATED SEARCH
# ─────────────────────────────────────────────────────────────

PAGE_SIZE = 12


def search_movies_df(
    df: pd.DataFrame,
    query: str,
    genre_filter: str  = "All",
    lang_filter:  str  = "All",
    year_range:   tuple = (1900, 2025),
    min_rating:   float = 0.0,
    content_type: str  = "All",
) -> pd.DataFrame:
    result = df.copy()
    if query and query.strip():
        q    = query.strip().lower()
        mask = result["primaryTitle"].str.lower().str.contains(q, na=False)
        result = result[mask]
    if genre_filter != "All":
        result = result[result["genres"].str.contains(genre_filter, na=False)]
    if lang_filter != "All":
        result = result[result["language"] == lang_filter]
    if content_type not in ("All", ""):
        type_map = {"Movies":"movie","TV Series":"tvseries"}
        if content_type in type_map:
            result = result[result["titleType"] == type_map[content_type]]
    result = result[
        result["startYear"].between(year_range[0], year_range[1]) &
        (result["averageRating"] >= min_rating)
    ]
    return result.sort_values("averageRating", ascending=False)


def render_search_bar(key_prefix: str = "search") -> str:
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] input {
        background:rgba(255,255,255,.06)!important;
        border:1.5px solid rgba(255,107,107,.35)!important;
        border-radius:10px!important;
        color:#e8e8e8!important;
        font-size:1rem!important;
        padding:.6rem 1rem!important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color:#FF6B6B!important;
        box-shadow:0 0 0 3px rgba(255,107,107,.15)!important;
    }
    </style>
    """, unsafe_allow_html=True)
    return st.text_input(
        "🔍 Search titles…",
        placeholder="e.g. Inception, The Dark Knight, Parasite",
        key=f"{key_prefix}_input",
        label_visibility="collapsed",
    )


def render_filter_row(df: pd.DataFrame, key_prefix: str = "filter") -> tuple:
    all_genres = sorted({
        g.strip()
        for genres in df["genres"].dropna()
        for g in genres.split(",")
        if g.strip()
    })
    all_langs = ["All"] + sorted(df["language"].dropna().unique().tolist())
    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
    with col1:
        genre = st.selectbox("Genre", ["All"] + all_genres, key=f"{key_prefix}_genre")
    with col2:
        lang  = st.selectbox("Language", all_langs, key=f"{key_prefix}_lang")
    with col3:
        min_r = st.number_input("Min ⭐", min_value=0.0, max_value=10.0,
                                value=6.0, step=0.5, format="%.1f", key=f"{key_prefix}_rating")
    with col4:
        cur_year  = int(df["startYear"].max()) if not df.empty else 2025
        year_range = st.slider("Year range", min_value=1900, max_value=cur_year,
                               value=(1990, cur_year), key=f"{key_prefix}_year")
    return genre, lang, min_r, year_range


def paginate(df: pd.DataFrame, key_prefix: str = "page") -> pd.DataFrame:
    total = len(df)
    if total == 0:
        return df
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    if f"{key_prefix}_num" not in st.session_state:
        st.session_state[f"{key_prefix}_num"] = 1
    page_num = max(1, min(st.session_state[f"{key_prefix}_num"], total_pages))
    start    = (page_num - 1) * PAGE_SIZE
    end      = start + PAGE_SIZE

    col_info, col_prev, col_jump, col_next = st.columns([3, 1, 1, 1])
    with col_info:
        st.markdown(
            f"<span style='color:#9ca3af;font-size:.85rem;'>"
            f"Showing {start+1}–{min(end,total)} of "
            f"<b style='color:#FF6B6B'>{total:,}</b> results</span>",
            unsafe_allow_html=True,
        )
    with col_prev:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=(page_num <= 1)):
            st.session_state[f"{key_prefix}_num"] = page_num - 1
            st.rerun()
    with col_jump:
        st.markdown(
            f"<div style='text-align:center;color:#e8e8e8;padding-top:.4rem'>"
            f"Page {page_num}/{total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=(page_num >= total_pages)):
            st.session_state[f"{key_prefix}_num"] = page_num + 1
            st.rerun()

    return df.iloc[start:end]


def reset_page(key_prefix: str = "page") -> None:
    st.session_state[f"{key_prefix}_num"] = 1


# ─────────────────────────────────────────────────────────────
# SECRETS VALIDATION
# ─────────────────────────────────────────────────────────────

def check_secrets() -> bool:
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
