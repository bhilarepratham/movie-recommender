"""
AI Agent for Enhanced Movie Recommendations
Uses natural language understanding to extract preferences and improve recommendations
"""
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class MovieAIAgent:
    """AI agent that understands natural language and extracts movie preferences"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.use_llm = True
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                self.use_llm = False
        else:
            self.use_llm = False
            self.logger.info("Using rule-based preference extraction (no OpenAI API key)")
        
        # Genre keywords mapping
        self.genre_keywords = {
            'Action': ['action', 'fight', 'combat', 'battle', 'war', 'explosive', 'thrilling'],
            'Comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious', 'comic'],
            'Drama': ['drama', 'emotional', 'serious', 'deep', 'touching'],
            'Romance': ['romance', 'romantic', 'love', 'dating', 'relationship'],
            'Horror': ['horror', 'scary', 'frightening', 'terrifying', 'spooky'],
            'Thriller': ['thriller', 'suspense', 'suspenseful', 'tense', 'edge'],
            'Sci-Fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien'],
            'Fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'dragon'],
            'Adventure': ['adventure', 'journey', 'quest', 'explore', 'expedition'],
            'Mystery': ['mystery', 'detective', 'investigation', 'puzzle', 'clue'],
            'Documentary': ['documentary', 'real', 'factual', 'educational'],
            'Animation': ['animation', 'animated', 'cartoon', 'anime'],
            'Crime': ['crime', 'criminal', 'gangster', 'heist', 'mafia'],
            'Biography': ['biography', 'biographical', 'true story', 'real person']
        }
        
        # Mood keywords
        self.mood_keywords = {
            'happy': ['happy', 'cheerful', 'uplifting', 'joyful', 'light'],
            'sad': ['sad', 'melancholic', 'depressing', 'emotional'],
            'excited': ['excited', 'energetic', 'pumped', 'adrenaline'],
            'relaxed': ['relaxed', 'calm', 'peaceful', 'chill', 'easy'],
            'thoughtful': ['thoughtful', 'deep', 'philosophical', 'meaningful'],
            'thrilled': ['thrilled', 'thrilling', 'intense', 'gripping']
        }
    
    def extract_preferences(self, user_query: str) -> Dict:
        """
        Extract movie preferences from natural language query
        
        Returns:
            Dictionary with extracted preferences:
            - genres: List of preferred genres
            - mood: Detected mood
            - min_rating: Minimum rating preference
            - year_range: Preferred year range
            - keywords: Important keywords
            - exclude_genres: Genres to avoid
        """
        query_lower = user_query.lower()
        preferences = {
            'genres': [],
            'mood': None,
            'min_rating': None,
            'year_range': None,
            'keywords': [],
            'exclude_genres': [],
            'directors': [],
            'actors': [],
            'language': None
        }
        
        if self.use_llm:
            return self._extract_with_llm(user_query, preferences)
        else:
            return self._extract_rule_based(query_lower, preferences)
    
    def _extract_with_llm(self, query: str, preferences: Dict) -> Dict:
        """Extract preferences using OpenAI API"""
        try:
            prompt = f"""Extract movie preferences from this user query: "{query}"

Return a JSON object with:
- genres: list of preferred genres (e.g., ["Action", "Comedy"])
- mood: one word mood (e.g., "happy", "thrilled", "thoughtful")
- min_rating: minimum IMDb rating (0-10) or null
- year_range: [start_year, end_year] or null
- keywords: important keywords from query
- exclude_genres: genres to avoid
- directors: mentioned director names
- actors: mentioned actor names
- language: preferred language or null

Only include fields that are clearly mentioned. Return valid JSON only."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a movie recommendation assistant. Extract preferences as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                preferences.update(extracted)
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}, falling back to rule-based")
            return self._extract_rule_based(query.lower(), preferences)
        
        return preferences
    
    def _extract_rule_based(self, query: str, preferences: Dict) -> Dict:
        """Extract preferences using rule-based approach"""
        # Extract genres
        for genre, keywords in self.genre_keywords.items():
            if any(kw in query for kw in keywords):
                if 'no ' + genre.lower() in query or 'not ' + genre.lower() in query:
                    preferences['exclude_genres'].append(genre)
                else:
                    preferences['genres'].append(genre)
        
        # Extract mood
        for mood, keywords in self.mood_keywords.items():
            if any(kw in query for kw in keywords):
                preferences['mood'] = mood
                break
        
        # Extract rating
        rating_match = re.search(r'rating[:\s]+(\d+\.?\d*)', query)
        if rating_match:
            preferences['min_rating'] = float(rating_match.group(1))
        elif 'high rating' in query or 'well rated' in query:
            preferences['min_rating'] = 7.5
        elif 'top rated' in query or 'best' in query:
            preferences['min_rating'] = 8.0
        
        # Extract year range
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if year_matches:
            years = [int(y) for y in year_matches]
            if len(years) >= 2:
                preferences['year_range'] = [min(years), max(years)]
            elif 'recent' in query or 'new' in query or 'latest' in query:
                preferences['year_range'] = [2015, 2024]
            elif 'old' in query or 'classic' in query or 'vintage' in query:
                preferences['year_range'] = [1900, 2000]
            else:
                year = years[0]
                preferences['year_range'] = [year - 5, year + 5]
        
        # Extract language
        languages = ['english', 'hindi', 'spanish', 'french', 'japanese', 'korean', 
                    'chinese', 'tamil', 'telugu', 'german', 'italian']
        for lang in languages:
            if lang in query:
                preferences['language'] = lang.capitalize()
                break
        
        # Extract keywords
        important_words = ['oscar', 'award', 'winning', 'popular', 'trending', 
                          'blockbuster', 'indie', 'independent', 'short', 'long']
        for word in important_words:
            if word in query:
                preferences['keywords'].append(word)
        
        return preferences
    
    def enhance_recommendations(self, recommendations: List[Dict], 
                               preferences: Dict, movies_df) -> List[Dict]:
        """
        Re-rank recommendations based on extracted preferences
        
        Args:
            recommendations: List of recommendation dictionaries
            preferences: Extracted preferences
            movies_df: DataFrame with movie data
        
        Returns:
            Re-ranked recommendations
        """
        scored_recs = []
        
        for rec in recommendations:
            score = rec.get('recommendation_score', 0)
            bonus = 0
            
            # Genre matching bonus
            if preferences.get('genres'):
                rec_genres = str(rec.get('genres', '')).lower()
                for pref_genre in preferences['genres']:
                    if pref_genre.lower() in rec_genres:
                        bonus += 2.0
            
            # Rating bonus
            if preferences.get('min_rating'):
                if rec.get('imdb_rating', 0) >= preferences['min_rating']:
                    bonus += 1.5
            
            # Year range bonus
            if preferences.get('year_range') and rec.get('year'):
                year_range = preferences['year_range']
                if year_range[0] <= rec['year'] <= year_range[1]:
                    bonus += 1.0
            
            # Language bonus
            if preferences.get('language'):
                movie_row = movies_df[movies_df['tconst'] == rec['movie_id']]
                if not movie_row.empty:
                    movie_lang = str(movie_row.iloc[0].get('language', '')).lower()
                    if preferences['language'].lower() in movie_lang:
                        bonus += 1.5
            
            # Penalty for excluded genres
            if preferences.get('exclude_genres'):
                rec_genres = str(rec.get('genres', '')).lower()
                for excl_genre in preferences['exclude_genres']:
                    if excl_genre.lower() in rec_genres:
                        bonus -= 3.0
            
            scored_recs.append({
                **rec,
                'ai_enhanced_score': score + bonus,
                'original_score': score
            })
        
        # Sort by enhanced score
        scored_recs.sort(key=lambda x: x['ai_enhanced_score'], reverse=True)
        
        return scored_recs
    
    def generate_response(self, user_query: str, recommendations: List[Dict]) -> str:
        """Generate a natural language response about recommendations"""
        if not recommendations:
            return "I couldn't find any movies matching your preferences. Try adjusting your criteria!"
        
        top_rec = recommendations[0]
        
        if self.use_llm:
            try:
                prompt = f"""User asked: "{user_query}"

Based on this, I'm recommending these movies:
{json.dumps(recommendations[:3], indent=2)}

Generate a friendly, conversational response (2-3 sentences) explaining why these recommendations match their query. Be enthusiastic but natural."""

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a friendly movie recommendation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"LLM response generation failed: {e}")
        
        # Fallback response
        genres = top_rec.get('genres', 'various genres')
        rating = top_rec.get('imdb_rating', 'N/A')
        
        return f"Based on your preferences, I found some great matches! Top recommendation: {top_rec['title']} ({top_rec.get('year', 'N/A')}) - a {genres} film rated {rating}/10 on IMDb. Check out the recommendations below!"
