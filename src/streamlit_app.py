"""
🎬 Complete Movie Recommendation Engine - Streamlit App
With working navigation, mood-based recommendations, and multilingual support
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# If we're in src directory, use it directly; otherwise look for src
if os.path.basename(current_dir) == 'src':
    src_dir = current_dir
else:
    src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from config import RecommenderConfig
from data_loader import MovieDataLoader
from preprocessor import DataPreprocessor
from movie_recommender import MovieRecommendationEngine
from ai_agent import MovieAIAgent

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    .main { 
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.85rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Enhanced Movie Cards */
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        margin: 1.2rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .movie-card:hover::before {
        left: 100%;
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    .movie-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 1.4em;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .movie-card p {
        margin: 0.5rem 0;
        font-size: 0.95em;
        opacity: 0.95;
    }
    
    /* User Greeting */
    .user-greeting {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        font-size: 2em;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
        position: relative;
        overflow: hidden;
    }
    .user-greeting::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    /* Streaming Badges */
    .streaming-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85em;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(76, 175, 80, 0.3);
        transition: all 0.3s;
    }
    .streaming-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.5);
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Info Boxes */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196F3;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4CAF50;
        border-radius: 10px;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chat Messages */
    .ai-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .user-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Mood Buttons */
    .mood-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .mood-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #333;
        font-weight: 700;
    }
    h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox and Multiselect */
    .stSelectbox>div>div, .stMultiSelect>div>div {
        border-radius: 10px;
    }
    
    /* Slider */
    .stSlider>div>div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Loading Spinner */
    .stSpinner>div {
        border-color: #667eea;
    }
    
    /* Quick Access Cards */
    .quick-access-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s;
        border: 2px solid transparent;
    }
    .quick-access-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"
if 'selected_mood' not in st.session_state:
    st.session_state.selected_mood = None
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = None

# Mood mapping
MOOD_MAPPING = {
    "😊 Happy": ["Comedy", "Family", "Musical", "Romance"],
    "😢 Sad": ["Drama"],
    "😱 Thrilled": ["Horror", "Thriller", "Action"],
    "🤔 Thoughtful": ["Drama", "Mystery", "Documentary"],
    "😂 Funny": ["Comedy"],
    "💪 Motivated": ["Biography", "Sport", "Adventure"],
    "🌙 Relaxed": ["Romance", "Family"],
    "🔥 Excited": ["Action", "Adventure", "Sci-Fi"]
}

# Cache model loading
@st.cache_resource
def load_model():
    """Load recommendation model"""
    with st.spinner('🎬 Loading...'):
        config = RecommenderConfig()
        loader = MovieDataLoader(config)
        loader.load_all_data()
        
        preprocessor = DataPreprocessor(config)
        train_df = loader.interactions_df
        train_df = preprocessor.filter_sparse_users_items(train_df)
        preprocessor.create_mappings(train_df)
        matrix = preprocessor.build_interaction_matrix(train_df)
        
        engine = MovieRecommendationEngine(config, preprocessor, loader)
        engine.train(matrix, show_progress=False)
        
        return engine, loader, preprocessor

# Load model
try:
    engine, loader, preprocessor = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    model_loaded = False

# Initialize AI Agent
if st.session_state.ai_agent is None:
    st.session_state.ai_agent = MovieAIAgent()

# Enhanced Login screen
if st.session_state.username is None:
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0;'>
            <h1 style='font-size: 3.5em; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                🎬 Movie Recommender
            </h1>
            <p style='font-size: 1.3em; color: #666; margin-bottom: 2rem;'>
                Discover Your Next Favorite Movie
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='background: white; padding: 2.5rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                <h2 style='text-align: center; color: #333; margin-bottom: 1.5rem;'>Welcome! 👤</h2>
                <p style='text-align: center; color: #666; margin-bottom: 2rem;'>Enter your name to get started</p>
            </div>
        """, unsafe_allow_html=True)
        
        username_input = st.text_input("Your Name:", placeholder="e.g., John Doe", key="login_input")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Let's Go!", type="primary", use_container_width=True):
                if username_input:
                    st.session_state.username = username_input
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter your name!")
    st.stop()

# Main app
st.markdown(f"""
    <div class="user-greeting">
        🎬 Welcome, {st.session_state.username}!
    </div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem 0; background: rgba(255,255,255,0.1); 
                    border-radius: 15px; margin-bottom: 1.5rem;'>
            <h3 style='color: white; margin: 0;'>👤 {st.session_state.username}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.username = None
        st.session_state.current_page = "🏠 Home"
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "🤖 AI Assistant", "🎭 Mood-Based", "🔍 Advanced Search", "🇮🇳 Indian Content", "📊 Browse All"],
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if model_loaded:
        st.markdown("### 📊 Statistics")
        st.metric("Total Movies", f"{len(loader.movies_df):,}")
        st.metric("Languages", loader.movies_df['language'].nunique())
        st.metric("Interactions", f"{len(loader.interactions_df):,}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='font-size: 2.5em; margin-bottom: 0.5rem;'>What would you like to watch?</h2>
            <p style='color: #666; font-size: 1.1em;'>Choose your mood and discover perfect movies</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Mood buttons
    st.markdown("### 🎭 Choose Your Mood")
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    moods = ["😊 Happy", "😢 Sad", "😱 Thrilled", "🤔 Thoughtful", 
             "😂 Funny", "💪 Motivated", "🌙 Relaxed", "🔥 Excited"]
    
    # Mood color gradients
    mood_colors = {
        "😊 Happy": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "😢 Sad": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "😱 Thrilled": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "🤔 Thoughtful": "linear-gradient(135deg, #30cfd0 0%, #330867 100%)",
        "😂 Funny": "linear-gradient(135deg, #fad961 0%, #f76b1c 100%)",
        "💪 Motivated": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
        "🌙 Relaxed": "linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)",
        "🔥 Excited": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    }
    
    for idx, mood in enumerate(moods):
        with cols[idx % 4]:
            mood_style = mood_colors.get(mood, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
            st.markdown(f"""
                <style>
                .mood-btn-{idx} {{
                    background: {mood_style};
                    color: white;
                    padding: 1.2rem;
                    border-radius: 15px;
                    text-align: center;
                    font-size: 1.3em;
                    font-weight: 700;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                    transition: all 0.3s ease;
                    cursor: pointer;
                    margin-bottom: 1rem;
                }}
                .mood-btn-{idx}:hover {{
                    transform: translateY(-5px) scale(1.05);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
                }}
                </style>
            """, unsafe_allow_html=True)
            
            if st.button(mood, key=f"mood_{idx}", use_container_width=True):
                st.session_state.selected_mood = mood
                
                # Show recommendations immediately
                st.markdown(f"""
                    <div style='text-align: center; margin: 2rem 0;'>
                        <h2 style='color: #667eea;'>{mood} Recommendations</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                sample_user = list(preprocessor.user_id_map.keys())[0]
                recs = engine.recommend(sample_user, n=6)
                
                rec_cols = st.columns(3)
                for i, rec in enumerate(recs[:6]):
                    with rec_cols[i % 3]:
                        platforms = ', '.join(rec['streaming_platforms'][:2])
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{rec['title']}</h4>
                            <p>⭐ {rec['imdb_rating']} | {rec['year']}</p>
                            <p>{rec['genres']}</p>
                            <p>📺 {platforms}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Quick filters
    st.markdown("### 🎯 Quick Access")
    
    quick_cols = st.columns(3)
    
    with quick_cols[0]:
        st.markdown("""
            <div class="quick-access-card">
                <h4 style='text-align: center; color: #667eea; margin-bottom: 1rem;'>🇮🇳 Indian Cinema</h4>
            </div>
        """, unsafe_allow_html=True)
        indian_count = len(loader.movies_df[loader.movies_df['language'].isin(['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada'])])
        st.metric("Available", f"{indian_count:,}")
        if st.button("Explore Indian Content", key="exp_indian", use_container_width=True):
            st.session_state.current_page = "🇮🇳 Indian Content"
            st.rerun()
    
    with quick_cols[1]:
        st.markdown("""
            <div class="quick-access-card">
                <h4 style='text-align: center; color: #667eea; margin-bottom: 1rem;'>🌏 Asian Content</h4>
            </div>
        """, unsafe_allow_html=True)
        asian_count = len(loader.movies_df[loader.movies_df['language'].isin(['Japanese', 'Korean', 'Chinese', 'Thai'])])
        st.metric("Available", f"{asian_count:,}")
    
    with quick_cols[2]:
        st.markdown("""
            <div class="quick-access-card">
                <h4 style='text-align: center; color: #667eea; margin-bottom: 1rem;'>🆕 Recent Releases</h4>
            </div>
        """, unsafe_allow_html=True)
        recent_count = len(loader.movies_df[loader.movies_df['startYear'] >= 2020])
        st.metric("Available", f"{recent_count:,}")
    
    st.markdown("---")
    
    # Enhanced Trending Section
    st.markdown("""
        <div style='text-align: center; margin: 3rem 0 2rem 0;'>
            <h2 style='font-size: 2.2em; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                🔥 Trending Now
            </h2>
            <p style='color: #666;'>Popular recommendations just for you</p>
        </div>
    """, unsafe_allow_html=True)
    
    if model_loaded:
        sample_user = list(preprocessor.user_id_map.keys())[0]
        trending = engine.recommend(sample_user, n=6)
        
        trend_cols = st.columns(3)
        for idx, rec in enumerate(trending):
            with trend_cols[idx % 3]:
                platforms = ' '.join([f'<span class="streaming-badge">{p}</span>' for p in rec['streaming_platforms'][:2]])
                st.markdown(f"""
                <div class="movie-card">
                    <h4>{rec['title']}</h4>
                    <p>⭐ {rec['imdb_rating']} | {rec['year']}</p>
                    <p>{rec['genres']}</p>
                    <div style='margin-top: 1rem;'>{platforms}</div>
                </div>
                """, unsafe_allow_html=True)

# AI ASSISTANT PAGE
elif page == "🤖 AI Assistant":
    st.markdown("## 🤖 AI Movie Assistant")
    st.markdown("**Chat with our AI to get personalized movie recommendations!**")
    
    # Show AI status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 Try: 'I want a thrilling action movie from the 2010s with high ratings' or 'Show me romantic comedies in Hindi'")
    with col2:
        ai_status = "🟢 Enhanced AI" if st.session_state.ai_agent.use_llm else "🟡 Rule-Based"
        st.markdown(f"**Status:** {ai_status}")
    
    # API Key input (optional)
    with st.expander("⚙️ Settings (Optional)"):
        api_key = st.text_input("OpenAI API Key (optional - for enhanced AI):", 
                                type="password", 
                                help="Enter your OpenAI API key for better AI understanding. Leave empty to use rule-based extraction.")
        if api_key and st.button("Save API Key"):
            st.session_state.ai_agent = MovieAIAgent(api_key=api_key)
            st.success("API key set! Enhanced AI features enabled.")
            st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Chat with AI")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.ai_chat_history):
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(msg['content'])
                    if 'recommendations' in msg and msg['recommendations']:
                        st.markdown("**🎬 Recommendations:**")
                        for rec in msg['recommendations'][:5]:
                            with st.expander(f"🎬 {rec['title']} ({rec.get('year', 'N/A')}) ⭐ {rec.get('imdb_rating', 'N/A')}"):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**Genres:** {rec.get('genres', 'N/A')}")
                                    st.markdown(f"**Director:** {rec.get('director', 'N/A')}")
                                    st.markdown(f"**Runtime:** {rec.get('runtime_minutes', 'N/A')} min")
                                    platforms = ', '.join(rec.get('streaming_platforms', ['Not Available'])[:3])
                                    st.markdown(f"**Streaming:** {platforms}")
                                with col2:
                                    if 'ai_enhanced_score' in rec:
                                        st.metric("AI Score", f"{rec['ai_enhanced_score']:.2f}")
                                    st.metric("IMDb", f"{rec.get('imdb_rating', 'N/A')}/10")
    
    # User input
    user_query = st.chat_input("Ask for movie recommendations...")
    
    if user_query:
        # Add user message to history
        st.session_state.ai_chat_history.append({'role': 'user', 'content': user_query})
        
        if model_loaded:
            with st.spinner("🤖 AI is thinking..."):
                # Extract preferences
                preferences = st.session_state.ai_agent.extract_preferences(user_query)
                
                # Get base recommendations
                sample_user = list(preprocessor.user_id_map.keys())[0]
                base_recs = engine.recommend(sample_user, n=20)
                
                # Enhance with AI
                enhanced_recs = st.session_state.ai_agent.enhance_recommendations(
                    base_recs, preferences, loader.movies_df
                )
                
                # Generate AI response
                ai_response = st.session_state.ai_agent.generate_response(user_query, enhanced_recs)
                
                # Add AI response to history
                st.session_state.ai_chat_history.append({
                    'role': 'assistant',
                    'content': ai_response,
                    'recommendations': enhanced_recs[:10]
                })
        else:
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': "Sorry, the recommendation model is not loaded. Please check the data files."
            })
        
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.ai_chat_history = []
        st.rerun()
    
    # Show extracted preferences (for debugging/transparency)
    if st.session_state.ai_chat_history:
        with st.expander("🔍 Last Query Analysis"):
            if model_loaded and st.session_state.ai_chat_history:
                last_user_msg = None
                for msg in reversed(st.session_state.ai_chat_history):
                    if msg['role'] == 'user':
                        last_user_msg = msg['content']
                        break
                
                if last_user_msg:
                    prefs = st.session_state.ai_agent.extract_preferences(last_user_msg)
                    st.json(prefs)

# MOOD-BASED PAGE
elif page == "🎭 Mood-Based":
    st.markdown("## 🎭 How are you feeling?")
    
    mood_selection = st.selectbox("Select your mood:", list(MOOD_MAPPING.keys()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_results = st.slider("Number of recommendations:", 5, 20, 10)
    
    with col2:
        platforms = st.multiselect("Streaming platforms:", 
                                   ["Netflix", "Amazon Prime", "Disney+", "Hulu", "HBO Max"])
    
    if st.button("🎬 Find Movies!", type="primary"):
        sample_user = list(preprocessor.user_id_map.keys())[0]
        
        filter_platforms = platforms if platforms else None
        recs = engine.recommend(sample_user, n=n_results * 2, filter_platforms=filter_platforms)
        
        # Filter by mood (genres)
        mood_genres = MOOD_MAPPING[mood_selection]
        filtered_recs = [r for r in recs if any(g in str(r['genres']) for g in mood_genres)][:n_results]
        
        if filtered_recs:
            st.success(f"✨ Found {len(filtered_recs)} perfect matches!")
            
            for idx, rec in enumerate(filtered_recs, 1):
                with st.expander(f"#{idx} - {rec['title']} ({rec['year']}) ⭐ {rec['imdb_rating']}", expanded=(idx <= 3)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Genres:** {rec['genres']}")
                        st.markdown(f"**Director:** {rec['director']}")
                        st.markdown(f"**Runtime:** {rec['runtime_minutes']} min")
                        
                        platforms_html = ' '.join([f'<span class="streaming-badge">{p}</span>' for p in rec['streaming_platforms']])
                        st.markdown(f"**Streaming:**", unsafe_allow_html=True)
                        st.markdown(platforms_html, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("IMDb", f"{rec['imdb_rating']}/10")
                        st.metric("Match", f"{abs(rec['recommendation_score']):.2f}")
        else:
            st.warning("No matches found. Try different filters!")

# INDIAN CONTENT PAGE
elif page == "🇮🇳 Indian Content":
    st.markdown("## 🇮🇳 Indian Language Cinema")
    
    indian_langs = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 'Bengali', 
                    'Marathi', 'Gujarati', 'Punjabi', 'Urdu']
    
    # Show available languages
    available_indian = [lang for lang in indian_langs if len(loader.movies_df[loader.movies_df['language'] == lang]) > 0]
    
    st.markdown(f"### Available Languages: {', '.join(available_indian)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_lang = st.selectbox("Select Language:", ["All Indian"] + available_indian)
    
    with col2:
        min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 6.0, 0.5)
    
    with col3:
        n_results = st.slider("Number of results:", 5, 30, 15)
    
    if st.button("🔍 Search", type="primary"):
        # Filter Indian content
        if selected_lang == "All Indian":
            filtered = loader.movies_df[loader.movies_df['language'].isin(available_indian)]
        else:
            filtered = loader.movies_df[loader.movies_df['language'] == selected_lang]
        
        filtered = filtered[filtered['averageRating'] >= min_rating]
        filtered = filtered.sort_values('averageRating', ascending=False).head(n_results)
        
        if len(filtered) > 0:
            st.success(f"✅ Found {len(filtered)} titles!")
            
            for idx, (_, movie) in enumerate(filtered.iterrows(), 1):
                with st.expander(f"#{idx} - {movie['primaryTitle']} ({int(movie['startYear'])}) - {movie.get('language', 'Unknown')}"):
                    st.markdown(f"**Language:** {movie.get('language', 'Unknown')}")
                    st.markdown(f"**IMDb Rating:** ⭐ {movie['averageRating']}/10")
                    st.markdown(f"**Genres:** {movie['genres']}")
                    st.markdown(f"**Runtime:** {movie['runtimeMinutes']} minutes")
        else:
            st.info(f"No {selected_lang} content found. Try running: get_real_languages.py")

# ADVANCED SEARCH PAGE
elif page == "🔍 Advanced Search":
    st.markdown("## 🔍 Advanced Search")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Language filter
        all_languages = sorted(loader.movies_df['language'].unique().tolist())
        selected_languages = st.multiselect("Languages:", all_languages)
        
        # Genre filter
        all_genres = set()
        for g in loader.movies_df['genres'].dropna():
            all_genres.update([x.strip() for x in str(g).split(',')])
        selected_genres = st.multiselect("Genres:", sorted(all_genres))
    
    with col2:
        # Rating
        min_rating = st.slider("Minimum IMDb Rating:", 0.0, 10.0, 6.0, 0.5)
        
        # Year
        year_min = int(loader.movies_df['startYear'].min())
        year_max = int(loader.movies_df['startYear'].max())
        year_range = st.slider("Year Range:", year_min, year_max, (2000, year_max))
    
    with col3:
        # Platforms
        platforms = st.multiselect("Streaming Platforms:", 
                                   ["Netflix", "Amazon Prime", "Disney+", "Hulu", "HBO Max", "Apple TV+"])
        
        # Results count
        n_results = st.slider("Number of results:", 5, 50, 20)
    
    if st.button("🔎 Search!", type="primary"):
        # Apply filters
        filtered = loader.movies_df.copy()
        
        if selected_languages:
            filtered = filtered[filtered['language'].isin(selected_languages)]
        
        if selected_genres:
            filtered = filtered[filtered['genres'].apply(
                lambda x: any(g in str(x) for g in selected_genres)
            )]
        
        filtered = filtered[filtered['averageRating'] >= min_rating]
        filtered = filtered[(filtered['startYear'] >= year_range[0]) & (filtered['startYear'] <= year_range[1])]
        
        # Sort and limit
        filtered = filtered.sort_values('averageRating', ascending=False).head(n_results)
        
        if len(filtered) > 0:
            st.success(f"✅ Found {len(filtered)} results!")
            
            for idx, (_, movie) in enumerate(filtered.iterrows(), 1):
                with st.expander(f"#{idx} - {movie['primaryTitle']} ({int(movie['startYear'])})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Language:** {movie.get('language', 'Unknown')}")
                        st.markdown(f"**Genres:** {movie['genres']}")
                        st.markdown(f"**Runtime:** {movie['runtimeMinutes']} min")
                    
                    with col2:
                        st.metric("IMDb", f"{movie['averageRating']}/10")
        else:
            st.warning("No results found. Try different filters!")

# BROWSE ALL PAGE
elif page == "📊 Browse All":
    st.markdown("## 📊 Browse All Content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox("Type:", ["All", "Movies", "TV Series"])
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Rating", "Year", "Title"])
    
    with col3:
        min_rating = st.slider("Min Rating:", 0.0, 10.0, 7.0)
    
    # Apply filters
    browse_df = loader.movies_df.copy()
    
    if filter_type == "Movies":
        browse_df = browse_df[browse_df['titleType'] == 'movie']
    elif filter_type == "TV Series":
        browse_df = browse_df[browse_df['titleType'] == 'tvSeries']
    
    browse_df = browse_df[browse_df['averageRating'] >= min_rating]
    
    if sort_by == "Rating":
        browse_df = browse_df.sort_values('averageRating', ascending=False)
    elif sort_by == "Year":
        browse_df = browse_df.sort_values('startYear', ascending=False)
    else:
        browse_df = browse_df.sort_values('primaryTitle')
    
    st.markdown(f"### Showing {len(browse_df):,} results")
    
    # Display in grid
    results_to_show = browse_df.head(24)
    cols = st.columns(4)
    
    for idx, (_, item) in enumerate(results_to_show.iterrows()):
        with cols[idx % 4]:
            type_emoji = "🎬" if item['titleType'] == 'movie' else "📺"
            st.markdown(f"""
            <div class="movie-card">
                <h5>{type_emoji} {item['primaryTitle']}</h5>
                <p>⭐ {item['averageRating']} | {int(item['startYear'])}</p>
                <p>{item.get('language', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                border-radius: 15px; margin-top: 2rem;'>
        <p style='font-size: 1.1em; color: #667eea; font-weight: 600; margin: 0.5rem 0;'>
            🎬 Personalized for {st.session_state.username}
        </p>
        <p style='color: #666; margin: 0.5rem 0;'>
            Total: {len(loader.movies_df):,} titles | Languages: {loader.movies_df['language'].nunique()} | 
            Interactions: {len(loader.interactions_df):,}
        </p>
        <p style='color: #999; font-size: 0.9em; margin-top: 1rem;'>
            Powered by AI-Powered Recommendation Engine
        </p>
    </div>
""", unsafe_allow_html=True)