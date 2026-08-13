import streamlit as st
import difflib
import PyPDF2
from gtts import gTTS
import tempfile
import pronouncing
import time
import os
import nltk
import re
import random
import threading
import json
from datetime import datetime
from pathlib import Path
from nltk.corpus import wordnet
from predefined_words import predefined_words

# Page Configuration
st.set_page_config(
    page_title="SpellBowl - Master Pronunciation & Spelling",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
LEADERBOARD_FILE = "leaderboard.json"
USERS_FILE = "users.json"

# Base directory of the app (used to find yearly word list folders regardless of cwd)
APP_DIR = Path(__file__).resolve().parent


# Yearly Prepopulated Word List Helpers
def get_available_years():
    """Find year folders (e.g. '2026') that sit next to spellbowl.py, newest first."""
    years = []
    if APP_DIR.exists():
        for entry in APP_DIR.iterdir():
            if entry.is_dir() and re.fullmatch(r"\d{4}", entry.name):
                years.append(entry.name)
    return sorted(years, reverse=True)


def get_categories_for_year(year):
    """Find division subfolders (e.g. 'elementary') inside a year folder that contain a PDF."""
    year_dir = APP_DIR / year
    categories = []
    if year_dir.exists():
        for entry in sorted(year_dir.iterdir()):
            if entry.is_dir() and any(entry.glob("*.pdf")):
                categories.append(entry.name)
    return categories


def get_pdf_path_for_category(year, category):
    """Return the first PDF found for a given year/division, or None."""
    cat_dir = APP_DIR / year / category
    pdfs = sorted(cat_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


# PDF Extraction Helpers

# PDF fonts commonly render these as single ligature glyphs instead of separate letters
# (e.g. "beneﬁting" instead of "benefiting"), which breaks naive letter-only regexes.
PDF_LIGATURES = {
    'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl',
    'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st',
}
# Smart/curly punctuation that word processors substitute for the plain ASCII versions
PDF_SMART_PUNCT = {
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '\xa0': ' ',
}


def normalize_pdf_text(text):
    """Undo common PDF text-extraction quirks (ligatures, smart quotes) that would
    otherwise fragment words like "o'clock" or "beneﬁting" during extraction."""
    for ligature, plain in PDF_LIGATURES.items():
        text = text.replace(ligature, plain)
    for fancy, plain in PDF_SMART_PUNCT.items():
        text = text.replace(fancy, plain)
    # Some PDFs render hyphenated compounds with a stray space before the hyphen
    # (e.g. "good -natured"); collapse that back into "good-natured".
    text = re.sub(r'([^\W\d_])\s+-\s*([^\W\d_])', r'\1-\2', text)
    return text


# Unicode-aware letter class (covers accented letters like "é") and word-char class
# (letters plus apostrophes/hyphens, for words like "o'clock" or "cross-cultural")
_LETTER = r'[^\W\d_]'
_WORDCHAR = r"(?:[^\W\d_]|['\-])"

# Matches numbered list entries like "1. aardvark", "2) Big Dipper", "3: about-face".
# Prefers stopping at the next list number or a newline, but falls back to a single
# word if no clean boundary is found (e.g. the last entry runs into trailing prose).
NUMBERED_ENTRY_RE = re.compile(
    r'\d+\s*[\.\)\:]\s*('
    rf'{_LETTER}{_WORDCHAR}*(?:[ \t]+{_LETTER}{_WORDCHAR}*){{0,3}}(?=\s*\d+\s*[\.\)\:]|\n|$)'
    r'|'
    rf'{_LETTER}{_WORDCHAR}*'
    r')'
)
MIN_NUMBERED_ENTRIES = 5  # below this, the PDF probably isn't a numbered list - fall back to full-text scan


def extract_numbered_entries(text):
    """Pull out only the words/phrases that follow a list number (e.g. '1. aardvark'),
    which skips titles, headers, and instructions on official word-list PDFs."""
    entries = []
    for raw in NUMBERED_ENTRY_RE.findall(text):
        entry = raw.strip()
        if entry and len(entry.replace(' ', '')) >= 2:
            entries.append(entry)
    return entries


def read_pdf_text(pdf_source):
    """Extract raw text from an uploaded file-like object or a PDF path on disk."""
    reader = PyPDF2.PdfReader(pdf_source)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"  # newline (not space) so page breaks don't glue words together
    return normalize_pdf_text(text)


@st.cache_resource(show_spinner="🧠 Loading smart extraction model (first time only)...")
def load_ner_pipeline():
    """Load a small Hugging Face NER model to help keep proper-noun phrases together.
    Returns None if transformers/torch aren't installed, so callers can fall back gracefully."""
    try:
        from transformers import pipeline
        return pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")
    except Exception:
        return None


def extract_ner_phrases(text, ner_pipe, chunk_chars=800):
    """Run NER over the text in chunks and return multi-word phrases like 'Mount Rushmore'."""
    phrases = set()
    words_in_text = text.split()

    chunks, chunk, chunk_len = [], [], 0
    for w in words_in_text:
        chunk.append(w)
        chunk_len += len(w) + 1
        if chunk_len >= chunk_chars:
            chunks.append(" ".join(chunk))
            chunk, chunk_len = [], 0
    if chunk:
        chunks.append(" ".join(chunk))

    for c in chunks:
        try:
            entities = ner_pipe(c)
        except Exception:
            continue
        for ent in entities:
            phrase = ent.get('word', '').strip()
            if phrase and not phrase.startswith('#') and len(phrase) >= 4:
                phrases.add(phrase)
    return phrases


def extract_words_from_text(text, use_smart_extraction=False):
    """Turn raw PDF text into a sorted, deduped word/phrase list (case preserved on first sighting).
    Prefers numbered-list entries ('1. aardvark') when the PDF looks like one, since that reliably
    skips titles/headers/instructions; otherwise falls back to scanning the whole text."""
    phrases = set()
    if use_smart_extraction:
        ner_pipe = load_ner_pipeline()
        if ner_pipe is not None:
            try:
                phrases = extract_ner_phrases(text, ner_pipe)
            except Exception as e:
                st.warning(f"⚠️ Smart extraction had an issue, using standard extraction instead. ({str(e)})")
        else:
            st.info("💡 Smart extraction needs extra packages (`pip install -r requirements-optional.txt`). Using standard extraction for now.")

    numbered_words = extract_numbered_entries(text)
    if len(numbered_words) >= MIN_NUMBERED_ENTRIES:
        words = numbered_words
    else:
        words = re.findall(rf'\b{_LETTER}{{4,}}\b', text)
        words = [w.strip() for w in words if len(w.strip()) >= 4]

    unique_words = {}
    for word in words:
        lower_word = word.lower()
        if lower_word not in unique_words:
            unique_words[lower_word] = word

    for phrase in phrases:
        phrase_clean = phrase.strip()
        if len(phrase_clean) < 4:
            continue
        lower_phrase = phrase_clean.lower()
        if lower_phrase not in unique_words:
            unique_words[lower_phrase] = phrase_clean
        for part in phrase_clean.split():
            unique_words.pop(part.lower(), None)

    return [unique_words[key] for key in sorted(unique_words.keys())]


# User Management Functions
def load_users():
    """Load users data from JSON file."""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}

def save_users(users):
    """Save users data to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving users: {e}")
        return False

def register_user(username, password, full_name):
    """Register a new user."""
    users = load_users()
    
    if username.lower() in [u.lower() for u in users.keys()]:
        return False, "Username already exists. Please choose another one."
    
    users[username] = {
        'password': password,  # In production, use hashed passwords!
        'full_name': full_name,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_quizzes': 0,
        'best_score': 0
    }
    
    if save_users(users):
        return True, "Registration successful!"
    return False, "Error during registration."

def authenticate_user(username, password):
    """Authenticate user login."""
    users = load_users()
    
    if username in users and users[username]['password'] == password:
        return True, users[username]['full_name']
    return False, None

def update_user_stats(username, accuracy):
    """Update user statistics after quiz completion."""
    users = load_users()
    
    if username in users:
        users[username]['total_quizzes'] = users[username].get('total_quizzes', 0) + 1
        users[username]['best_score'] = max(users[username].get('best_score', 0), accuracy)
        save_users(users)

def load_leaderboard():
    """Load leaderboard data from JSON file."""
    try:
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r') as f:
                data = json.load(f)
                # Handle old format (list) and convert to new format (dict)
                if isinstance(data, list):
                    # Convert old list format to new dict format
                    new_format = {}
                    for entry in data:
                        username = entry['name']
                        if username not in new_format:
                            new_format[username] = {
                                'name': entry['name'],
                                'total_score': entry['score'],
                                'total_questions': entry['total'],
                                'total_quizzes': 1,
                                'best_accuracy': entry['accuracy'],
                                'avg_accuracy': entry['accuracy'],
                                'last_quiz_date': entry.get('timestamp', entry.get('date', '')),
                                'quiz_history': [{
                                    'score': entry['score'],
                                    'total': entry['total'],
                                    'accuracy': entry['accuracy'],
                                    'timestamp': entry.get('timestamp', ''),
                                    'word_source': entry.get('word_source', 'unknown')
                                }]
                            }
                        else:
                            # Aggregate multiple entries for same user
                            new_format[username]['total_score'] += entry['score']
                            new_format[username]['total_questions'] += entry['total']
                            new_format[username]['total_quizzes'] += 1
                            new_format[username]['best_accuracy'] = max(new_format[username]['best_accuracy'], entry['accuracy'])
                            new_format[username]['last_quiz_date'] = entry.get('timestamp', entry.get('date', ''))
                            new_format[username]['quiz_history'].append({
                                'score': entry['score'],
                                'total': entry['total'],
                                'accuracy': entry['accuracy'],
                                'timestamp': entry.get('timestamp', ''),
                                'word_source': entry.get('word_source', 'unknown')
                            })
                            # Recalculate average accuracy
                            total_acc = sum(q['accuracy'] for q in new_format[username]['quiz_history'])
                            new_format[username]['avg_accuracy'] = round(total_acc / len(new_format[username]['quiz_history']), 1)
                    return new_format
                return data
        return {}
    except Exception as e:
        st.error(f"Error loading leaderboard: {e}")
        return {}

def save_to_leaderboard(name, score, total, accuracy, word_source, total_words):
    """Save a quiz result to the leaderboard - updates existing user or creates new entry."""
    try:
        leaderboard = load_leaderboard()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        quiz_entry = {
            'score': score,
            'total': total,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'word_source': word_source
        }
        
        if name in leaderboard:
            # Update existing user
            leaderboard[name]['total_score'] += score
            leaderboard[name]['total_questions'] += total
            leaderboard[name]['total_quizzes'] += 1
            leaderboard[name]['best_accuracy'] = max(leaderboard[name]['best_accuracy'], accuracy)
            leaderboard[name]['last_quiz_date'] = timestamp
            leaderboard[name]['quiz_history'].append(quiz_entry)
            
            # Recalculate average accuracy
            total_acc = sum(q['accuracy'] for q in leaderboard[name]['quiz_history'])
            leaderboard[name]['avg_accuracy'] = round(total_acc / len(leaderboard[name]['quiz_history']), 1)
        else:
            # Create new user entry
            leaderboard[name] = {
                'name': name,
                'total_score': score,
                'total_questions': total,
                'total_quizzes': 1,
                'best_accuracy': accuracy,
                'avg_accuracy': accuracy,
                'last_quiz_date': timestamp,
                'quiz_history': [quiz_entry]
            }
        
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(leaderboard, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving to leaderboard: {e}")
        return False

def get_top_scores(limit=10):
    """Get top scores from leaderboard sorted by best accuracy."""
    leaderboard = load_leaderboard()
    # Convert dict to list and sort by best accuracy (descending), then by total score
    leaderboard_list = list(leaderboard.values())
    sorted_board = sorted(leaderboard_list, key=lambda x: (x['best_accuracy'], x['total_score']), reverse=True)
    return sorted_board[:limit]

def get_user_scores(name, limit=5):
    """Get recent quiz history for a specific user."""
    leaderboard = load_leaderboard()
    if name in leaderboard:
        user_data = leaderboard[name]
        # Return most recent quizzes (reversed to show newest first)
        recent_quizzes = user_data['quiz_history'][-limit:][::-1]
        return user_data, recent_quizzes
    return None, []

def sidebar_leaderboard():
    """Display leaderboard in sidebar."""
    with st.sidebar:
        st.markdown("### 🏆 Leaderboard")
        st.markdown("---")
        
        # Get top 5 scores for sidebar
        top_scores = get_top_scores(5)
        
        if top_scores:
            for idx, entry in enumerate(top_scores, 1):
                if idx == 1:
                    medal = "🥇"
                elif idx == 2:
                    medal = "🥈"
                elif idx == 3:
                    medal = "🥉"
                else:
                    medal = f"#{idx}"
                
                st.markdown(f"""
                <div style='background: #f8f9fa; 
                            padding: 0.5em; 
                            border-radius: 8px; 
                            margin: 0.3em 0;
                            border-left: 3px solid #667eea;'>
                    <p style='margin: 0; font-size: 0.9em; font-weight: 600; color: #2a3b5d;'>
                        {medal} {entry['name']}
                    </p>
                    <p style='margin: 0.2em 0 0 0; font-size: 0.75em; color: #636e72;'>
                        Best: {entry['best_accuracy']}% | Avg: {entry['avg_accuracy']}%
                    </p>
                    <p style='margin: 0.2em 0 0 0; font-size: 0.7em; color: #95a5a6;'>
                        {entry['total_quizzes']} quiz(es) | {entry['total_score']}/{entry['total_questions']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🎯 No scores yet!")
        
        st.markdown("---")

def play_audio(text, rate=100):
    """Play text using Google TTS with specified speech rate."""
    import time
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Add timeout and slow parameter
            tts = gTTS(text=text, lang='en', slow=(rate < 80), timeout=10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # Verify file was created and has content
            if not Path(temp_file).exists() or Path(temp_file).stat().st_size == 0:
                raise Exception("Audio file not created properly")
            
            # Display audio player with autoplay enabled
            try:
                with open(temp_file, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    
                if len(audio_bytes) > 0:
                    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                    st.success("🔊 Audio ready! Tap play button if it doesn't start automatically.", icon="✅")
                else:
                    raise Exception("Empty audio file")
                    
            except Exception as audio_error:
                st.warning(f"⚠️ Audio playback issue: {str(audio_error)}")
                st.info("💡 Tap the play button on the audio player above to hear the word.")
            
            # Schedule cleanup after a delay (async-like)
            try:
                import threading
                threading.Timer(15.0, lambda: Path(temp_file).unlink(missing_ok=True)).start()
            except Exception:
                # If threading fails, try immediate cleanup after a short delay
                try:
                    time.sleep(1)
                    Path(temp_file).unlink(missing_ok=True)
                except Exception:
                    pass
            
            # Success - break out of retry loop
            break
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            if retry_count < max_retries:
                st.warning(f"⚠️ Attempt {retry_count} failed. Retrying... ({error_msg})")
                time.sleep(1)  # Wait before retry
            else:
                st.error(f"❌ Could not generate audio after {max_retries} attempts.")
                st.error(f"**Error details:** {error_msg}")
                
                # Provide detailed troubleshooting
                with st.expander("� Troubleshooting Steps", expanded=True):
                    st.markdown("""
                    ### Possible Issues:
                    
                    1. **Network Connection:**
                       - Check if you have a stable internet connection
                       - Try switching between WiFi and mobile data
                       - Google TTS requires internet to generate audio
                    
                    2. **Browser Issues:**
                       - Try refreshing the page (swipe down to refresh)
                       - Clear browser cache
                       - Try a different browser (Chrome, Firefox, Safari)
                    
                    3. **Mobile Settings:**
                       - Disable Low Power Mode (affects network performance)
                       - Disable Data Saver mode
                       - Check if the site has permission to use network
                    
                    4. **Firewall/Network Restrictions:**
                       - Check if your network blocks Google TTS API
                       - Try using a different network
                    
                    5. **Alternative:**
                       - Try using the "Manual Word Pronunciation" tab
                       - Type the word manually to hear it
                    """)
                break

@st.cache_resource
def load_word_list():
    """Load NLTK word list once and cache it."""
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words
    return set(w.lower() for w in nltk_words.words())

@st.cache_resource
def load_wordnet():
    """Load WordNet data once."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    return True

@st.cache_data
def get_word_info(word):
    """Get word definition, synonyms, and antonyms using WordNet."""
    load_wordnet()
    
    try:
        synsets = wordnet.synsets(word.lower())
        
        if not synsets:
            return {
                'meaning': None,
                'synonym': [],
                'antonym': []
            }
        
        # Get definitions grouped by part of speech
        meanings = {}
        all_synonyms = set()
        all_antonyms = set()
        
        for synset in synsets:
            # Get part of speech
            pos = synset.pos()
            pos_name = {
                'n': 'Noun',
                'v': 'Verb',
                'a': 'Adjective',
                's': 'Adjective Satellite',
                'r': 'Adverb'
            }.get(pos, 'Other')
            
            # Add definition
            if pos_name not in meanings:
                meanings[pos_name] = []
            meanings[pos_name].append(synset.definition())
            
            # Get synonyms (lemmas)
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    all_synonyms.add(synonym)
                
                # Get antonyms
                for antonym in lemma.antonyms():
                    all_antonyms.add(antonym.name().replace('_', ' '))
        
        return {
            'meaning': meanings if meanings else None,
            'synonym': list(all_synonyms),
            'antonym': list(all_antonyms)
        }
    except Exception as e:
        return {
            'meaning': None,
            'synonym': [],
            'antonym': [],
            'error': str(e)
        }

@st.cache_data
def get_system_generated_words(level):
    """Get system generated words based on difficulty level using NLTK corpus."""
    # Load NLTK word list
    word_list = load_word_list()
    
    # Filter words by length and complexity for each level
    if level == 'Level 1 (Grade 1-3)':
        # Simple 3-5 letter common words
        filtered_words = [
            word for word in word_list 
            if 3 <= len(word) <= 5 
            and word.isalpha() 
            and word.islower()
        ]
    elif level == 'Level 2 (Grade 4-6)':
        # Medium 5-7 letter words
        filtered_words = [
            word for word in word_list 
            if 5 <= len(word) <= 7 
            and word.isalpha() 
            and word.islower()
        ]
    elif level == 'Level 3 (Grade 7-10)':
        # Advanced 7-10 letter words
        filtered_words = [
            word for word in word_list 
            if 7 <= len(word) <= 10 
            and word.isalpha() 
            and word.islower()
        ]
    else:  # Level 4 (Grade 10-12)
        # Complex 10+ letter words
        filtered_words = [
            word for word in word_list 
            if len(word) >= 10 
            and word.isalpha() 
            and word.islower()
        ]
    
    # Return random selection of filtered words
    import random
    if len(filtered_words) > 500:
        return random.sample(filtered_words, 500)
    return list(filtered_words)

def quiz_tile(speech_rate=100):
    """Interactive pronunciation quiz tile."""
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">🎯 Pronunciation Quiz</div>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'student_name' not in st.session_state:
            st.session_state.student_name = ""
        if 'name_submitted' not in st.session_state:
            st.session_state.name_submitted = False
        if 'quiz_words' not in st.session_state:
            st.session_state.quiz_words = []
        if 'used_quiz_words' not in st.session_state:
            st.session_state.used_quiz_words = []
        if 'current_quiz_word' not in st.session_state:
            st.session_state.current_quiz_word = None
        if 'quiz_attempts' not in st.session_state:
            st.session_state.quiz_attempts = 0
        if 'quiz_score' not in st.session_state:
            st.session_state.quiz_score = 0
        if 'quiz_total' not in st.session_state:
            st.session_state.quiz_total = 0
        if 'answer_submitted' not in st.session_state:
            st.session_state.answer_submitted = False
        if 'wrong_attempts' not in st.session_state:
            st.session_state.wrong_attempts = []  # Track wrong spelling attempts
        if 'quiz_history' not in st.session_state:
            st.session_state.quiz_history = []  # Track performance history (1 for correct, 0 for wrong)
        if 'competition_mode' not in st.session_state:
            st.session_state.competition_mode = False
        if 'timer_seconds' not in st.session_state:
            st.session_state.timer_seconds = 30
        if 'timer_start' not in st.session_state:
            st.session_state.timer_start = None
        if 'time_expired' not in st.session_state:
            st.session_state.time_expired = False
        
        # Login/Registration section
        if not st.session_state.name_submitted:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2em; 
                        border-radius: 15px; 
                        margin: 1em 0;
                        text-align: center;
                        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);'>
                <p style='margin: 0; font-size: 2.5em;'>👋</p>
                <p style='margin: 0.5em 0 0 0; color: white; font-size: 1.8em; font-weight: 700;'>
                    Welcome to SpellBowl!
                </p>
                <p style='margin: 0.3em 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.1em;'>
                    Login or Register to start your pronunciation journey
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize auth mode in session state
            if 'auth_mode' not in st.session_state:
                st.session_state.auth_mode = 'login'
            
            # Toggle between login and register
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                if st.button("🔑 Login", use_container_width=True, type="primary" if st.session_state.auth_mode == 'login' else "secondary"):
                    st.session_state.auth_mode = 'login'
                    st.rerun()
            with col_tab2:
                if st.button("📝 Register", use_container_width=True, type="primary" if st.session_state.auth_mode == 'register' else "secondary"):
                    st.session_state.auth_mode = 'register'
                    st.rerun()
            
            st.markdown("---")
            
            # Login Form
            if st.session_state.auth_mode == 'login':
                with st.form(key="login_form"):
                    st.markdown("### 🔑 Login to Your Account")
                    
                    username = st.text_input(
                        "Username",
                        placeholder="Enter your username",
                        key="login_username"
                    )
                    
                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="login_password"
                    )
                    
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        login_button = st.form_submit_button("🚀 Login", use_container_width=True)
                    
                    if login_button:
                        if username and password:
                            success, full_name = authenticate_user(username, password)
                            if success:
                                st.session_state.student_name = full_name
                                st.session_state.username = username
                                st.session_state.name_submitted = True
                                
                                # Get user stats
                                users = load_users()
                                user_data = users.get(username, {})
                                st.success(f"🎉 Welcome back, {full_name}! 🌟")
                                st.info(f"📊 Your Stats: {user_data.get('total_quizzes', 0)} quiz(es) completed | Best Score: {user_data.get('best_score', 0)}%")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid username or password. Please try again.")
                        else:
                            st.warning("⚠️ Please enter both username and password.")
            
            # Registration Form
            else:
                with st.form(key="register_form"):
                    st.markdown("### 📝 Create New Account")
                    
                    full_name = st.text_input(
                        "Full Name",
                        placeholder="Enter your full name",
                        key="register_fullname"
                    )
                    
                    username = st.text_input(
                        "Username",
                        placeholder="Choose a unique username",
                        key="register_username",
                        help="Username must be unique"
                    )
                    
                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Choose a strong password",
                        key="register_password",
                        help="Minimum 4 characters recommended"
                    )
                    
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        placeholder="Re-enter your password",
                        key="register_confirm_password"
                    )
                    
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        register_button = st.form_submit_button("✨ Create Account", use_container_width=True)
                    
                    if register_button:
                        if full_name and username and password and confirm_password:
                            if password != confirm_password:
                                st.error("❌ Passwords do not match!")
                            elif len(password) < 4:
                                st.error("❌ Password must be at least 4 characters long!")
                            elif len(username) < 3:
                                st.error("❌ Username must be at least 3 characters long!")
                            else:
                                success, message = register_user(username, password, full_name)
                                if success:
                                    st.success(f"✅ {message} You can now login!")
                                    st.balloons()
                                    time.sleep(2)
                                    st.session_state.auth_mode = 'login'
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.warning("⚠️ Please fill in all fields.")
            
            st.markdown("---")
            st.info("💡 **Tip:** Login or register above to start the quiz and track your progress!")
            return  # Stop here until logged in
        
        # Show personalized greeting after login
        col_greeting, col_logout = st.columns([4, 1])
        with col_greeting:
            users = load_users()
            username = st.session_state.get('username', '')
            user_data = users.get(username, {})
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1em; 
                        border-radius: 10px; 
                        margin-bottom: 1em;
                        border-left: 5px solid #667eea;'>
                <p style='margin: 0; color: #2a3b5d; font-size: 1.2em; font-weight: 600;'>
                    👤 Hello, <strong>{st.session_state.student_name}</strong>! (@{username})
                </p>
                <p style='margin: 0.3em 0 0 0; color: #636e72; font-size: 0.9em;'>
                    📊 {user_data.get('total_quizzes', 0)} quizzes | ⭐ Best: {user_data.get('best_score', 0)}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_logout:
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                st.session_state.name_submitted = False
                st.session_state.student_name = ""
                st.session_state.username = ""
                st.session_state.auth_mode = 'login'
                st.success("👋 Logged out successfully!")
                time.sleep(1)
                st.rerun()
        
        # Competition Mode Settings
        st.markdown("---")
        st.markdown("### ⚡ Competition Mode")
        
        col_mode, col_timer = st.columns([2, 1])
        
        with col_mode:
            competition_enabled = st.checkbox(
                "🏆 Enable Competition Mode",
                value=st.session_state.competition_mode,
                key="competition_mode_checkbox",
                help="In Competition Mode, you must answer each question within the time limit!"
            )
            st.session_state.competition_mode = competition_enabled
            
            if competition_enabled:
                st.info("⏱️ **Competition Mode Active:** Answer each question before time runs out!")
        
        with col_timer:
            if competition_enabled:
                timer_seconds = st.number_input(
                    "⏱️ Time per question (seconds)",
                    min_value=5,
                    max_value=120,
                    value=st.session_state.timer_seconds,
                    step=5,
                    key="timer_input",
                    help="Set how many seconds you have to answer each question"
                )
                st.session_state.timer_seconds = timer_seconds
        
        # Word source selection
        st.markdown("---")
        st.markdown("### 📚 Choose Word Source")
        st.caption("Not sure which to pick? Try **📅 This Year's Word List** — it's already loaded for you, no file needed!")
        word_source = st.radio(
            "Select word source:",
            options=["Predefined Source", "📅 This Year's Word List", "Upload PDF", "System Generated"],
            key="word_source_radio",
            horizontal=True
        )

        # Initialize quiz_pdf / selected_pdf_path to None
        quiz_pdf = None
        selected_pdf_path = None

        if word_source == "Predefined Source":
            # Use predefined words array
            st.info("📚 Using predefined word list with 750 curated words")
            
            # Load button for predefined words
            if st.button("📥 Load Predefined Words", key="load_predefined_words_btn"):
                try:
                    # Use the imported predefined_words array
                    all_words = predefined_words.copy()
                    
                    if not all_words:
                        st.error("No words found in predefined list.")
                    else:
                        # Store all words and select first 50 by default
                        st.session_state.all_loaded_words = all_words
                        st.session_state.quiz_words = all_words[:50] if len(all_words) > 50 else all_words
                        st.session_state.word_source_type = "predefined"
                        
                        # Reset quiz state when new words are loaded
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.wrong_attempts = []
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.session_state.last_pdf_name = "predefined_list"
                        
                        st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words from predefined list!")
                        st.info("👇 Select word range below and click 'Get Random Word' to start!")
                except Exception as e:
                    st.error(f"Error loading predefined words: {str(e)}")
                
        elif word_source == "📅 This Year's Word List":
            # Teacher-provided, prepopulated word lists organized by year/division
            available_years = get_available_years()
            if not available_years:
                st.warning("📭 No yearly word lists have been added yet. Ask your teacher, or pick another word source!")
            else:
                col_year, col_division = st.columns(2)
                with col_year:
                    selected_year = st.selectbox("📆 Year", options=available_years, key="year_select")

                categories = get_categories_for_year(selected_year)
                if not categories:
                    st.warning(f"📭 No word lists found for {selected_year} yet.")
                else:
                    with col_division:
                        selected_category = st.selectbox(
                            "🏫 Division",
                            options=categories,
                            format_func=lambda c: c.replace('_', ' ').title(),
                            key=f"category_select_{selected_year}"
                        )
                    pdf_path = get_pdf_path_for_category(selected_year, selected_category)
                    if pdf_path:
                        selected_pdf_path = pdf_path
                        st.success(f"📖 Using the **{selected_category.title()}** list for **{selected_year}** ({pdf_path.name})")
                    else:
                        st.warning("📭 No PDF found for this division yet.")

        elif word_source == "Upload PDF":
            # PDF uploader for quiz words
            st.caption("💡 Ask a parent or teacher to help you find the right PDF file if you're not sure!")
            quiz_pdf = st.file_uploader("Upload PDF for quiz words", type=["pdf"], key="quiz_pdf_uploader")
        else:
            # System generated word selection
            difficulty_level = st.selectbox(
                "Select Difficulty Level:",
                options=[
                    'Level 1 (Grade 1-3)',
                    'Level 2 (Grade 4-6)',
                    'Level 3 (Grade 7-10)',
                    'Level 4 (Grade 10-12)'
                ],
                key="difficulty_level_select"
            )
            
            # Button to load system generated words
            if st.button("📥 Load System Words", key="load_system_words_btn"):
                system_words = get_system_generated_words(difficulty_level)
                import random
                st.session_state.all_loaded_words = random.sample(system_words, min(500, len(system_words)))
                st.session_state.quiz_words = st.session_state.all_loaded_words[:50]  # Default to first 50
                st.session_state.word_source_type = "system"
                
                # Reset quiz state when new words are loaded
                st.session_state.used_quiz_words = []
                st.session_state.current_quiz_word = None
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.answer_submitted = False
                st.session_state.wrong_attempts = []
                st.session_state.quiz_history = []
                st.session_state.last_pdf_name = None  # Clear PDF tracking
                st.session_state.leaderboard_saved = False
                
                st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words from {difficulty_level}!")
                st.info("👇 Select word range below and click 'Get Random Word' to start!")
                st.rerun()

        # Optional smart extraction toggle for PDF-based sources
        use_smart_extraction = False
        if word_source in ("Upload PDF", "📅 This Year's Word List") and (quiz_pdf is not None or selected_pdf_path is not None):
            use_smart_extraction = st.checkbox(
                "✨ Smart extraction (keeps proper-noun phrases like \"Mount Rushmore\" together)",
                value=False,
                key="use_smart_extraction",
                help="Uses a small AI model (Hugging Face) to detect multi-word names. Downloads ~250MB the first time it's used."
            )

        # Check if a PDF (uploaded or a prepopulated yearly one) needs processing
        pdf_source = quiz_pdf if quiz_pdf is not None else selected_pdf_path
        pdf_identifier = quiz_pdf.name if quiz_pdf is not None else (str(selected_pdf_path) if selected_pdf_path else None)

        if pdf_source is not None and (not st.session_state.quiz_words or st.session_state.get('last_pdf_name') != pdf_identifier):
            try:
                with st.spinner("📖 Reading your PDF... this'll just take a moment!"):
                    text = read_pdf_text(pdf_source)
                    all_words = extract_words_from_text(text, use_smart_extraction=use_smart_extraction)

                if not all_words:
                    st.error("No valid words found in PDF. Please upload a different PDF.")
                    st.info("💡 Make sure your PDF contains readable text (not scanned images).")
                    return

                # Store all words and select first 50 by default
                st.session_state.all_loaded_words = all_words
                st.session_state.quiz_words = all_words[:50] if len(all_words) > 50 else all_words
                st.session_state.word_source_type = "yearly" if selected_pdf_path is not None else "pdf"

                # Reset quiz state when new PDF is loaded
                st.session_state.used_quiz_words = []
                st.session_state.current_quiz_word = None
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.answer_submitted = False
                st.session_state.wrong_attempts = []
                st.session_state.quiz_history = []
                st.session_state.last_pdf_name = pdf_identifier
                st.session_state.leaderboard_saved = False

                st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words!")
                st.info("👇 Select word range below and click 'Get Random Word' to start!")
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return
        
        if st.session_state.quiz_words:
            # Word range selector
            if 'all_loaded_words' in st.session_state and len(st.session_state.all_loaded_words) > 0:
                st.markdown("### 🎯 Select Word Range")
                
                total_words = len(st.session_state.all_loaded_words)
                
                # Create preset range buttons
                col_range1, col_range2, col_range3, col_range4, col_range5 = st.columns(5)
                
                with col_range1:
                    if st.button("📘 1-10", key="range_1_10", use_container_width=True):
                        st.session_state.quiz_words = st.session_state.all_loaded_words[:10]
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.rerun()
                
                with col_range2:
                    if st.button("📗 11-25", key="range_11_25", use_container_width=True):
                        st.session_state.quiz_words = st.session_state.all_loaded_words[10:25]
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.rerun()
                
                with col_range3:
                    if st.button("📙 26-50", key="range_26_50", use_container_width=True):
                        st.session_state.quiz_words = st.session_state.all_loaded_words[25:50]
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.rerun()
                
                with col_range4:
                    if st.button("📕 51-100", key="range_51_100", use_container_width=True):
                        st.session_state.quiz_words = st.session_state.all_loaded_words[50:100]
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.rerun()
                
                with col_range5:
                    if st.button("📚 All Words", key="range_all", use_container_width=True):
                        st.session_state.quiz_words = st.session_state.all_loaded_words
                        st.session_state.used_quiz_words = []
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_total = 0
                        st.session_state.answer_submitted = False
                        st.session_state.quiz_history = []
                        st.session_state.leaderboard_saved = False
                        st.rerun()
                
                # Custom range selector
                st.markdown("**Or select custom range:**")
                col_custom1, col_custom2, col_custom3 = st.columns([2, 2, 1])
                
                with col_custom1:
                    start_range = st.number_input(
                        "Start word #", 
                        min_value=1, 
                        max_value=total_words, 
                        value=1,
                        key="start_range_input"
                    )
                
                with col_custom2:
                    end_range = st.number_input(
                        "End word #", 
                        min_value=1, 
                        max_value=total_words, 
                        value=min(50, total_words),
                        key="end_range_input"
                    )
                
                with col_custom3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✅ Apply", key="apply_custom_range", use_container_width=True):
                        if start_range <= end_range:
                            st.session_state.quiz_words = st.session_state.all_loaded_words[start_range-1:end_range]
                            st.session_state.used_quiz_words = []
                            st.session_state.current_quiz_word = None
                            st.session_state.quiz_score = 0
                            st.session_state.quiz_total = 0
                            st.session_state.answer_submitted = False
                            st.session_state.quiz_history = []
                            st.session_state.leaderboard_saved = False
                            st.success(f"✅ Selected words {start_range} to {end_range} ({end_range - start_range + 1} words)")
                            st.rerun()
                        else:
                            st.error("Start word must be less than or equal to end word!")
                
                # Show current range details
                if st.session_state.quiz_words and 'all_loaded_words' in st.session_state:
                    first_word_idx = st.session_state.all_loaded_words.index(st.session_state.quiz_words[0]) + 1 if st.session_state.quiz_words else 0
                    last_word_idx = st.session_state.all_loaded_words.index(st.session_state.quiz_words[-1]) + 1 if st.session_state.quiz_words else 0
                    st.info(f"📊 Currently practicing: **{len(st.session_state.quiz_words)} words** (Word #{first_word_idx} to #{last_word_idx} from total **{total_words} words**)")
                else:
                    st.info(f"📊 Currently practicing: **{len(st.session_state.quiz_words)} words** from total **{total_words} words**")
                st.markdown("---")
            
            # Display score
            remaining = len(st.session_state.quiz_words) - len(st.session_state.used_quiz_words)
            col_score1, col_score2, col_score3 = st.columns(3)
            
            with col_score1:
                st.metric("Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
            with col_score2:
                if st.session_state.quiz_total > 0:
                    percentage = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
                    st.metric("Accuracy", f"{percentage:.1f}%")
                else:
                    st.metric("Accuracy", "0%")
            with col_score3:
                st.metric("Remaining Words", remaining)
            
            # Performance Visualization
            if st.session_state.quiz_history:
                st.markdown("### 📊 Performance Tracker")
                
                # Create visual progress bar with emojis and word details
                history_display = ""
                for idx, result in enumerate(st.session_state.quiz_history, 1):
                    # Get the word for this question
                    word_idx = idx - 1
                    if word_idx < len(st.session_state.used_quiz_words):
                        word = st.session_state.used_quiz_words[word_idx]
                        if result == 1:
                            history_display += f'<span style="color: #10b981; font-weight: 600;" title="✅ {word}">✅</span> '
                        else:
                            history_display += f'<span style="color: #ef4444; font-weight: 600;" title="❌ {word}">❌</span> '
                    else:
                        if result == 1:
                            history_display += "✅ "
                        else:
                            history_display += "❌ "
                    
                    # Add line break every 10 results
                    if idx % 10 == 0:
                        history_display += "<br>"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                            padding: 1em; 
                            border-radius: 10px; 
                            border-left: 4px solid #0ea5e9; 
                            margin: 1em 0;
                            text-align: center;'>
                    <p style='margin: 0; color: #0c4a6e; font-size: 1.2em; line-height: 1.8;'>
                        {history_display}
                    </p>
                    <p style='margin: 0.5em 0 0 0; color: #64748b; font-size: 0.85em; font-style: italic;'>
                        💡 Hover over each emoji to see the word
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detailed list in expander
                with st.expander("📋 View Detailed Performance", expanded=False):
                    for idx, result in enumerate(st.session_state.quiz_history, 1):
                        word_idx = idx - 1
                        if word_idx < len(st.session_state.used_quiz_words):
                            word = st.session_state.used_quiz_words[word_idx]
                            if result == 1:
                                st.markdown(f"**{idx}.** ✅ **{word.upper()}** - Correct")
                            else:
                                st.markdown(f"**{idx}.** ❌ **{word.upper()}** - Wrong")
                
                # Show streak information
                if st.session_state.quiz_history:
                    current_streak = 0
                    max_streak = 0
                    temp_streak = 0
                    
                    for result in reversed(st.session_state.quiz_history):
                        if result == 1:
                            if current_streak == temp_streak:
                                current_streak += 1
                            temp_streak += 1
                            max_streak = max(max_streak, temp_streak)
                        else:
                            temp_streak = 0
                    
                    col_streak1, col_streak2 = st.columns(2)
                    with col_streak1:
                        st.metric("🔥 Current Streak", f"{current_streak}")
                    with col_streak2:
                        st.metric("⭐ Best Streak", f"{max_streak}")
            
            # Show status message
            if st.session_state.current_quiz_word is None:
                st.info(f"👉 {st.session_state.student_name}, click 'Get Random Word' to start!")
            elif not st.session_state.answer_submitted:
                st.info(f"🎧 {st.session_state.student_name}, listen carefully and type your answer!")
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            
            with col_a:
                # Check if all words have been used
                if st.session_state.quiz_words:
                    available_words = [w for w in st.session_state.quiz_words if w not in st.session_state.used_quiz_words]
                    
                    if available_words:
                        # Only allow getting a new word if no current word or answer was already submitted
                        can_get_word = st.session_state.current_quiz_word is None or st.session_state.answer_submitted
                        
                        if can_get_word:
                            if st.button("🎲 Get Random Word", key="random_word_btn"):
                                import random
                                selected_word = random.choice(available_words)
                                st.session_state.current_quiz_word = selected_word
                                st.session_state.quiz_attempts = 0
                                st.session_state.answer_submitted = False
                                st.session_state.time_expired = False
                                # Reset timer - will start when pronunciation is played
                                st.session_state.timer_start = None
                                
                                # Debug: Find the word position in original list
                                if 'all_loaded_words' in st.session_state:
                                    try:
                                        word_position = st.session_state.all_loaded_words.index(selected_word) + 1
                                        st.toast(f"Word #{word_position} selected from full list!", icon="✅")
                                    except ValueError:
                                        st.toast(f"Word selected! Click 'Play Pronunciation' to hear it.", icon="✅")
                                else:
                                    st.toast(f"Word selected! Click 'Play Pronunciation' to hear it.", icon="✅")
                                st.rerun()
                        else:
                            st.button("🎲 Get Random Word", key="random_word_btn", disabled=True)
                            st.caption("⚠️ Answer the current word first or skip it")
                    else:
                        # Quiz completed - save to leaderboard
                        final_accuracy = (st.session_state.quiz_score / st.session_state.quiz_total * 100) if st.session_state.quiz_total > 0 else 0
                        
                        # Save to leaderboard if not already saved
                        if 'leaderboard_saved' not in st.session_state or not st.session_state.leaderboard_saved:
                            word_source = st.session_state.get('word_source_type', 'unknown')
                            total_words = len(st.session_state.get('all_loaded_words', []))
                            
                            if save_to_leaderboard(
                                st.session_state.student_name,
                                st.session_state.quiz_score,
                                st.session_state.quiz_total,
                                round(final_accuracy, 1),
                                word_source,
                                total_words
                            ):
                                st.session_state.leaderboard_saved = True
                                # Update user stats
                                if 'username' in st.session_state:
                                    update_user_stats(st.session_state.username, round(final_accuracy, 1))
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                                    padding: 2em; 
                                    border-radius: 15px; 
                                    text-align: center;
                                    border: 3px solid #fdcb6e;
                                    margin: 1em 0;'>
                            <p style='margin: 0; font-size: 3em;'>🥳</p>
                            <p style='margin: 0.5em 0 0 0; color: #2d3436; font-size: 1.8em; font-weight: 800;'>
                                Congratulations, {st.session_state.student_name}!
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #2d3436; font-size: 1.2em;'>
                                You've completed the quiz! 🌟
                            </p>
                            <p style='margin: 0.5em 0 0 0; color: #636e72; font-size: 1em;'>
                                Final Score: {st.session_state.quiz_score}/{st.session_state.quiz_total} ({final_accuracy:.1f}%)
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #2ecc71; font-size: 0.9em; font-weight: 600;'>
                                ✅ Score saved to leaderboard!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🔄 Reset Quiz", key="reset_quiz_btn"):
                            st.session_state.used_quiz_words = []
                            st.session_state.current_quiz_word = None
                            st.session_state.quiz_score = 0
                            st.session_state.quiz_total = 0
                            st.session_state.answer_submitted = False
                            st.session_state.quiz_history = []
                            st.session_state.leaderboard_saved = False
                            st.rerun()
                else:
                    st.warning("Please upload a PDF first to load quiz words.")
            
            with col_b:
                if st.session_state.current_quiz_word:
                    if st.button("🔊 Play Pronunciation", key="quiz_play_btn"):
                        with st.spinner("🎵 Generating audio... Please wait..."):
                            play_audio(st.session_state.current_quiz_word, rate=speech_rate)
                        # Start timer after audio plays (only if competition mode and not already started)
                        if st.session_state.competition_mode and st.session_state.timer_start is None and not st.session_state.answer_submitted:
                            st.session_state.timer_start = time.time()
                            st.rerun()
                else:
                    st.button("🔊 Play Pronunciation", key="quiz_play_btn", disabled=True)
            
            with col_c:
                if st.session_state.current_quiz_word:
                    if st.button("⏭️ Skip Word", key="skip_word_btn"):
                        if not st.session_state.answer_submitted:
                            st.session_state.used_quiz_words.append(st.session_state.current_quiz_word)
                            st.session_state.current_quiz_word = None
                            st.session_state.timer_start = None
                            st.session_state.time_expired = False
                            st.rerun()
                else:
                    st.button("⏭️ Skip Word", key="skip_word_btn", disabled=True)
            
            # Show the quiz interface when a word is selected
            if st.session_state.current_quiz_word:
                st.markdown("---")
                
                # Competition Mode Timer Display or Waiting Message
                if st.session_state.competition_mode and not st.session_state.answer_submitted:
                    if st.session_state.timer_start is None:
                        # Timer not started yet - waiting for audio to play
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                                    padding: 1em; 
                                    border-radius: 10px; 
                                    border: 3px solid #f59e0b; 
                                    margin: 1em 0;
                                    text-align: center;'>
                            <p style='margin: 0; color: #92400e; font-size: 1.2em; font-weight: 700;'>
                                ⚡ Competition Mode Active
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #78350f; font-size: 0.9em;'>
                                Click "Play Pronunciation" to start the timer!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Timer is running
                        elapsed_time = time.time() - st.session_state.timer_start
                        remaining_time = max(0, st.session_state.timer_seconds - elapsed_time)
                        
                        if remaining_time > 0:
                            # Calculate color based on remaining time
                            time_percentage = (remaining_time / st.session_state.timer_seconds) * 100
                            if time_percentage > 50:
                                timer_color = "#10b981"  # Green
                            elif time_percentage > 25:
                                timer_color = "#f59e0b"  # Orange
                            else:
                                timer_color = "#ef4444"  # Red
                            
                            # Create placeholder for timer that will be updated
                            timer_placeholder = st.empty()
                            timer_placeholder.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                        padding: 1em; 
                                        border-radius: 10px; 
                                        border: 3px solid {timer_color}; 
                                        margin: 1em 0;
                                        text-align: center;'>
                                <p style='margin: 0; color: {timer_color}; font-size: 2.5em; font-weight: 800;'>
                                    ⏱️ {int(remaining_time)}s
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #0c4a6e; font-size: 0.9em;'>
                                    Time Remaining
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Time expired
                            if not st.session_state.time_expired:
                                st.session_state.time_expired = True
                                st.session_state.answer_submitted = True
                                st.session_state.quiz_total += 1
                                st.session_state.quiz_history.append(0)
                                st.session_state.used_quiz_words.append(st.session_state.current_quiz_word)
                                
                                st.session_state.wrong_attempts.append({
                                    'correct': st.session_state.current_quiz_word,
                                    'your_answer': '(Time Expired)',
                                    'similarity': 0,
                                    'error_type': 'timeout'
                                })
                            
                            st.error("⏰ **TIME'S UP!** You ran out of time for this question.")
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                        padding: 1.5em; 
                                        border-radius: 12px; 
                                        border-left: 5px solid #10b981; 
                                        margin: 1em 0;
                                        text-align: center;'>
                                <p style='margin: 0; color: #065f46; font-size: 0.9em; font-weight: 600; text-transform: uppercase;'>
                                    The correct answer was:
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #047857; font-size: 2.5em; font-weight: 800;'>
                                    {st.session_state.current_quiz_word}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.info(f"🎧 {st.session_state.student_name}, listen to the pronunciation and spell the word below:")
                st.success(f"✓ Word selected! ({len(st.session_state.current_quiz_word)} letters)")
                
                # Add Hint Button
                if not st.session_state.answer_submitted:
                    if st.button("💡 Get Hint", key="hint_btn"):
                        word_info = get_word_info(st.session_state.current_quiz_word)
                        
                        with st.expander("📖 Word Hints", expanded=True):
                            st.markdown(f"### Hints for the word ({len(st.session_state.current_quiz_word)} letters)")
                            
                            # Show definition
                            if word_info['meaning']:
                                st.markdown("#### 📚 Definition:")
                                for part_of_speech, definitions in word_info['meaning'].items():
                                    st.markdown(f"**{part_of_speech.capitalize()}:**")
                                    for idx, definition in enumerate(definitions[:2], 1):  # Show first 2 definitions
                                        st.write(f"{idx}. {definition}")
                            else:
                                st.warning("Definition not available for this word.")
                            
                            # Show synonyms
                            if word_info['synonym']:
                                st.markdown("#### 🔄 Synonyms:")
                                synonyms_list = word_info['synonym'][:5]  # Show first 5 synonyms
                                st.write(", ".join(synonyms_list))
                            
                            # Show antonyms
                            if word_info['antonym']:
                                st.markdown("#### ↔️ Antonyms:")
                                antonyms_list = word_info['antonym'][:5]  # Show first 5 antonyms
                                st.write(", ".join(antonyms_list))
                
                is_answer_submitted = bool(st.session_state.answer_submitted)
                
                # Create a form to allow Enter key submission
                with st.form(key="answer_form", clear_on_submit=False):
                    user_answer = st.text_input("Your spelling:", key="quiz_answer_input", disabled=is_answer_submitted, placeholder="Type the word you heard and press Enter...")
                    submit_button = st.form_submit_button("Check Answer", disabled=is_answer_submitted)
                
                if submit_button and not is_answer_submitted:
                    if user_answer:
                        correct_word = st.session_state.current_quiz_word
                        st.session_state.quiz_attempts += 1
                        st.session_state.quiz_total += 1
                        st.session_state.answer_submitted = True
                        st.session_state.used_quiz_words.append(correct_word)
                        
                        if user_answer == correct_word:
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_history.append(1)  # Track correct answer
                            
                            # Animated success message
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                        padding: 2em; 
                                        border-radius: 15px; 
                                        border: 3px solid #10b981; 
                                        margin: 1em 0;
                                        text-align: center;
                                        animation: successPulse 0.5s ease-in-out;'>
                                <p style='margin: 0; font-size: 3em;'>🎉</p>
                                <p style='margin: 0.5em 0 0 0; color: #065f46; font-size: 1.5em; font-weight: 800;'>
                                    CORRECT!
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #047857; font-size: 1.2em; font-weight: 600;'>
                                    The word is: {word}
                                </p>
                            </div>
                            <style>
                            @keyframes successPulse {{
                                0% {{ transform: scale(0.8); opacity: 0; }}
                                50% {{ transform: scale(1.05); }}
                                100% {{ transform: scale(1); opacity: 1; }}
                            }}
                            </style>
                            """.replace('{word}', correct_word.upper()), unsafe_allow_html=True)
                            
                            phones = pronouncing.phones_for_word(correct_word)
                            if phones:
                                st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                            
                            # Celebration animation for correct answer
                            st.balloons()
                        else:
                            similarity = difflib.SequenceMatcher(None, user_answer, correct_word).ratio()
                            st.session_state.quiz_history.append(0)  # Track wrong answer
                            
                            # Determine what went wrong: case or spelling or both
                            case_mismatch = user_answer.lower() == correct_word.lower() and user_answer != correct_word
                            spelling_wrong = user_answer.lower() != correct_word.lower()
                            
                            if case_mismatch:
                                error_type = "❗ Case Sensitivity Error"
                                error_detail = "Your spelling is correct, but the capitalization is wrong!"
                                error_icon = "🔡"
                            elif spelling_wrong:
                                error_type = "❌ Spelling Error"
                                error_detail = "The spelling is incorrect."
                                error_icon = "❌"
                            
                            # Track wrong attempt for revision
                            st.session_state.wrong_attempts.append({
                                'correct': correct_word,
                                'your_answer': user_answer,
                                'similarity': similarity * 100,
                                'error_type': 'case' if case_mismatch else 'spelling'
                            })
                            
                            # Animated error message
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                        padding: 2em; 
                                        border-radius: 15px; 
                                        border: 3px solid #ef4444; 
                                        margin: 1em 0;
                                        text-align: center;
                                        animation: shakeTilt 0.5s ease-in-out;'>
                                <p style='margin: 0; font-size: 3em;'>{error_icon}</p>
                                <p style='margin: 0.5em 0 0 0; color: #991b1b; font-size: 1.5em; font-weight: 800;'>
                                    {error_type}
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #b91c1c; font-size: 1em;'>
                                    {error_detail}
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #b91c1c; font-size: 1em;'>
                                    You wrote: <strong>{user_answer}</strong>
                                </p>
                            </div>
                            <style>
                            @keyframes shakeTilt {{
                                0%, 100% {{ transform: translateX(0) rotate(0deg); }}
                                25% {{ transform: translateX(-5px) rotate(-2deg); }}
                                75% {{ transform: translateX(5px) rotate(2deg); }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Show correct spelling prominently with actual case
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                        padding: 1.5em; 
                                        border-radius: 12px; 
                                        border-left: 5px solid #10b981; 
                                        margin: 1em 0;
                                        text-align: center;
                                        animation: slideIn 0.5s ease-out;'>
                                <p style='margin: 0; color: #065f46; font-size: 0.9em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>
                                    Correct Spelling
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #047857; font-size: 2.5em; font-weight: 800; letter-spacing: 0.02em;'>
                                    {correct_word}
                                </p>
                            </div>
                            <style>
                            @keyframes slideIn {{
                                from {{ transform: translateY(-20px); opacity: 0; }}
                                to {{ transform: translateY(0); opacity: 1; }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Snow animation for wrong answer
                            st.snow()
                            
                            st.info(f"📊 Similarity Score: **{similarity*100:.1f}%** - You were {similarity*100:.1f}% close!")
                            
                            phones = pronouncing.phones_for_word(correct_word)
                            if phones:
                                st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                            
                            # Give hints based on error type and similarity
                            if case_mismatch:
                                st.warning(f"⚠️ **Case Sensitivity Tip:** Pay attention to which letters are uppercase and lowercase. The correct word is: **{correct_word}**")
                            elif similarity > 0.7:
                                st.info("💡 You're very close! Just a few letters off.")
                            elif similarity > 0.5:
                                st.info(f"💡 The word has {len(correct_word)} letters and you got most of them right.")
                            else:
                                st.info(f"💡 Tip: The word starts with **'{correct_word[0]}'** and has **{len(correct_word)} letters**.")
                    else:
                        st.warning("Please enter a word before checking.")
                
                # Show result after answer is submitted
                if st.session_state.answer_submitted:
                    correct_word = st.session_state.current_quiz_word
                    
                    col_next1, col_next2 = st.columns(2)
                    with col_next1:
                        if st.button("➡️ Next Word", key="next_word_btn", use_container_width=True):
                            st.session_state.current_quiz_word = None
                            st.session_state.answer_submitted = False
                            st.session_state.timer_start = None
                            st.session_state.time_expired = False
                            st.rerun()
                    
                    with col_next2:
                        if st.button("🔊 Hear it again", key="hear_again_btn", use_container_width=True):
                            play_audio(correct_word, rate=speech_rate)
            
            # Show revision list of wrong attempts
            if st.session_state.wrong_attempts:
                st.markdown("---")
                st.markdown("### 📝 Revision List - Words to Practice")
                st.info(f"You have **{len(st.session_state.wrong_attempts)}** word(s) to review")
                
                with st.expander("View All Wrong Attempts", expanded=False):
                    for idx, attempt in enumerate(st.session_state.wrong_attempts, 1):
                        if attempt.get('error_type') == 'timeout':
                            error_badge = "⏰ Time Expired"
                            border_color = "#ef4444"
                        elif attempt.get('error_type') == 'case':
                            error_badge = "🔡 Case Error"
                            border_color = "#3b82f6"
                        else:
                            error_badge = "❌ Spelling Error"
                            border_color = "#f59e0b"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                                    padding: 1em; 
                                    border-radius: 10px; 
                                    border-left: 4px solid {border_color}; 
                                    margin: 0.8em 0;'>
                            <p style='margin: 0; color: #92400e; font-size: 0.85em; font-weight: 600;'>
                                #{idx} - {error_badge} - Similarity: {attempt['similarity']:.1f}%
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #b45309;'>
                                <strong>Your Answer:</strong> <span style='text-decoration: line-through;'>{attempt['your_answer']}</span>
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #065f46; font-weight: 700;'>
                                <strong>Correct:</strong> {attempt['correct']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_rev1, col_rev2 = st.columns([1, 3])
                        with col_rev1:
                            if st.button(f"🔊 Hear", key=f"revision_play_{idx}"):
                                play_audio(attempt['correct'], rate=speech_rate)
                
                if st.button("🗑️ Clear Revision List", key="clear_revision_btn"):
                    st.session_state.wrong_attempts = []
                    st.rerun()
            
            # Auto-refresh for timer updates (only if timer is running and no answer submitted)
            if (st.session_state.competition_mode and 
                st.session_state.current_quiz_word and 
                st.session_state.timer_start and 
                not st.session_state.answer_submitted):
                elapsed = time.time() - st.session_state.timer_start
                if elapsed < st.session_state.timer_seconds:
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("📤 Choose a word source above to start the pronunciation quiz!")
            st.markdown("""
            **How it works:**
            1. Pick a word source: the predefined list, this year's official list, your own PDF, or a difficulty level
            2. System will load words for you (up to 50 to start)
            3. Click 'Get Random Word' to start
            4. Listen to pronunciation (no spelling shown!)
            5. Type what you heard and check your answer
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def leaderboard_tile():
    """Display leaderboard with top scores and user history."""
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">🏆 Leaderboard</div>', unsafe_allow_html=True)
        
        # Create tabs for different leaderboard views
        lb_tab1, lb_tab2, lb_tab3 = st.tabs(["🌟 Top Performers", "👤 My Scores", "📊 All Scores"])
        
        with lb_tab1:
            st.markdown("### 🏅 Top 10 Performers")
            st.markdown("*Ranked by best accuracy*")
            
            top_scores = get_top_scores(10)
            
            if top_scores:
                for idx, entry in enumerate(top_scores, 1):
                    # Medal emojis for top 3
                    if idx == 1:
                        medal = "🥇"
                        bg_color = "#ffd700"
                    elif idx == 2:
                        medal = "🥈"
                        bg_color = "#c0c0c0"
                    elif idx == 3:
                        medal = "🥉"
                        bg_color = "#cd7f32"
                    else:
                        medal = f"#{idx}"
                        bg_color = "#e3eafc"
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {bg_color}20 0%, {bg_color}10 100%); 
                                padding: 1em; 
                                border-radius: 10px; 
                                border-left: 4px solid {bg_color}; 
                                margin: 0.5em 0;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <p style='margin: 0; font-size: 1.2em; font-weight: 700; color: #2a3b5d;'>
                                    {medal} {entry['name']}
                                </p>
                                <p style='margin: 0.2em 0 0 0; font-size: 0.85em; color: #636e72;'>
                                    📅 Last Quiz: {entry['last_quiz_date']} | 🎯 {entry['total_quizzes']} quiz(es)
                                </p>
                            </div>
                            <div style='text-align: right;'>
                                <p style='margin: 0; font-size: 1.5em; font-weight: 800; color: #2ecc71;'>
                                    {entry['best_accuracy']}%
                                </p>
                                <p style='margin: 0; font-size: 0.9em; color: #636e72;'>
                                    Best | Avg: {entry['avg_accuracy']}%
                                </p>
                                <p style='margin: 0; font-size: 0.85em; color: #95a5a6;'>
                                    {entry['total_score']}/{entry['total_questions']} total
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🎯 No scores yet! Complete a quiz to be the first on the leaderboard!")
        
        with lb_tab2:
            st.markdown("### 📈 My Recent Performance")
            
            if st.session_state.get('name_submitted') and st.session_state.get('student_name'):
                user_name = st.session_state.student_name
                user_data, user_scores = get_user_scores(user_name, 10)
                
                if user_data:
                    st.success(f"Found {user_data['total_quizzes']} quiz(es) for **{user_name}**")
                    
                    # Display stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Total Quizzes", user_data['total_quizzes'])
                    with col2:
                        st.metric("📊 Avg Accuracy", f"{user_data['avg_accuracy']}%")
                    with col3:
                        st.metric("⭐ Best Score", f"{user_data['best_accuracy']}%")
                    with col4:
                        st.metric("📝 Total Score", f"{user_data['total_score']}/{user_data['total_questions']}")
                    
                    st.markdown("---")
                    st.markdown("**Recent Attempts:**")
                    
                    for idx, entry in enumerate(user_scores, 1):
                        # Determine color based on accuracy
                        if entry['accuracy'] >= 90:
                            color = "#2ecc71"
                            emoji = "🌟"
                        elif entry['accuracy'] >= 75:
                            color = "#3498db"
                            emoji = "👍"
                        elif entry['accuracy'] >= 50:
                            color = "#f39c12"
                            emoji = "📖"
                        else:
                            color = "#e74c3c"
                            emoji = "💪"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                                    padding: 0.8em; 
                                    border-radius: 8px; 
                                    border-left: 3px solid {color}; 
                                    margin: 0.4em 0;'>
                            <p style='margin: 0; font-size: 0.95em; font-weight: 600; color: #2a3b5d;'>
                                {emoji} Quiz #{user_data['total_quizzes'] - idx + 1} - {entry['timestamp']}
                            </p>
                            <p style='margin: 0.2em 0 0 0; font-size: 0.85em; color: #636e72;'>
                                Score: <strong>{entry['score']}/{entry['total']}</strong> | 
                                Accuracy: <strong style='color: {color};'>{entry['accuracy']}%</strong> | 
                                Source: {entry['word_source'].title()}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No scores found for **{user_name}**. Complete a quiz to see your scores here!")
            else:
                st.warning("👤 Please login to view your scores!")
        
        with lb_tab3:
            st.markdown("### 📋 All Users Overview")
            
            all_scores = load_leaderboard()
            
            if all_scores:
                st.info(f"Showing {len(all_scores)} users")
                
                # Convert dict to list and sort by best accuracy
                all_users = list(all_scores.values())
                all_users_sorted = sorted(all_users, key=lambda x: (x['best_accuracy'], x['total_score']), reverse=True)
                
                # Display in a scrollable container
                for idx, entry in enumerate(all_users_sorted, 1):
                    accuracy_color = "#2ecc71" if entry['best_accuracy'] >= 75 else "#f39c12" if entry['best_accuracy'] >= 50 else "#e74c3c"
                    
                    st.markdown(f"""
                    <div style='background: #f8f9fa; 
                                padding: 0.7em; 
                                border-radius: 6px; 
                                margin: 0.3em 0;
                                border-left: 3px solid {accuracy_color};'>
                        <p style='margin: 0; font-size: 0.9em; color: #2a3b5d;'>
                            <strong>#{idx} {entry['name']}</strong>
                        </p>
                        <p style='margin: 0.2em 0 0 0; font-size: 0.8em; color: #636e72;'>
                            Best: {entry['best_accuracy']}% | Avg: {entry['avg_accuracy']}% | 
                            {entry['total_quizzes']} quiz(es) | Total: {entry['total_score']}/{entry['total_questions']}
                        </p>
                        <p style='margin: 0.2em 0 0 0; font-size: 0.75em; color: #95a5a6;'>
                            Last quiz: {entry['last_quiz_date']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No scores recorded yet. Be the first to complete a quiz!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def spelling_checker_tile(speech_rate=100):
    """Spelling checker tile with pronunciation helper."""
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">🔤 Spelling Checker & Pronunciation Helper</div>', unsafe_allow_html=True)
        spelling_input = st.text_input("Enter a word to check spelling:", key="spelling_input")
        if spelling_input:
            word_list = load_word_list()
            if spelling_input.lower() in word_list:
                st.success("Spelling is correct!")
                phones = pronouncing.phones_for_word(spelling_input.lower())
                if phones:
                    st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                    if st.button("Speak Spelling Word", key="spelling_speak_btn"):
                        play_audio(spelling_input, rate=speech_rate)
                else:
                    st.info("Pronunciation not found for this word.")
            else:
                suggestions = difflib.get_close_matches(spelling_input.lower(), word_list, n=1, cutoff=0.7)
                if suggestions:
                    correct_word = suggestions[0]
                    similarity = difflib.SequenceMatcher(None, spelling_input.lower(), correct_word).ratio()
                    st.error(f"Spelling is incorrect. Did you mean: **{correct_word}**?")
                    st.info(f"Similarity: {similarity*100:.1f}%")
                    phones = pronouncing.phones_for_word(correct_word)
                    if phones:
                        st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                        if st.button("Speak Correct Word", key="correct_speak_btn"):
                            play_audio(correct_word, rate=speech_rate)
                    else:
                        st.info("Pronunciation not found for the correct word.")
                else:
                    st.error("Spelling is incorrect and no close match found.")
        st.markdown('</div>', unsafe_allow_html=True)



st.markdown("""
<style>
.tile {
    background: linear-gradient(135deg, #e3eafc 0%, #f8f9fa 100%);
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border: 1px solid #d1d9e6;
    padding: 1.7em 1.3em 1.3em 1.3em;
    margin-bottom: 2em;
    transition: box-shadow 0.2s;
}
.tile:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.tile-title {
    font-size: 1.35em;
    font-weight: bold;
    margin-bottom: 0.7em;
    color: #2a3b5d;
}
.pronunciation {
    font-size: 1.12em;
    color: #0072e3;
    font-weight: bold;
}
.speak-btn {
    background: #0072e3;
    color: white;
    border-radius: 6px;
    padding: 0.5em 1.2em;
    font-size: 1em;
    border: none;
    margin-top: 0.5em;
}
</style>
""", unsafe_allow_html=True)


st.title("🗣️ Word Pronunciation Helper")

# Disclaimer and Credits
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1.2em; 
            border-radius: 10px; 
            margin-bottom: 1.5em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
    <div style='text-align: center; color: white;'>
        <p style='margin: 0; font-size: 0.9em; font-weight: 600; opacity: 0.95;'>
            ✨ Made with ❤️ for MRE ✨
        </p>
        <p style='margin: 0.3em 0 0 0; font-size: 0.85em; opacity: 0.85;'>
            <strong>Aashrita Choudhari</strong> & <strong>Rahul Choudhari</strong>
        </p>
        <p style='margin: 0.5em 0 0 0; font-size: 0.75em; opacity: 0.75; font-style: italic;'>
            Empowering students to master pronunciation and spelling
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Mobile audio notice
with st.expander("📱 Using on Mobile? Read this!", expanded=False):
    st.markdown("""
    ### 🔊 Audio Settings for iPhone/iPad:
    
    **If audio is not playing automatically:**
    1. **Unmute your device** - Check the mute switch on the side of your iPhone
    2. **Turn up volume** - Use volume buttons to increase sound
    3. **Tap the play button** - You may need to manually tap the audio player that appears
    4. **Enable autoplay in Safari:**
       - Go to Settings → Safari → Website Settings
       - Find "Auto-Play" and set to "Allow All Auto-Play"
    5. **Try Chrome or Firefox** - Sometimes works better than Safari on iOS
    
    **Note:** iOS devices may require user interaction before audio can play automatically. 
    After clicking "Play Pronunciation", look for the audio player below and tap it if needed.
    """)

rate_slider = st.slider("Adjust speech rate (%)", min_value=30, max_value=150, value=100, step=10, key="speech_rate_slider")

# Display sidebar leaderboard
sidebar_leaderboard()

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Pronunciation Quiz",
    "📝 Spelling Checker & Pronunciation Helper",
    "📄 PDF Word Pronunciation",
    "✍️ Manual Word Pronunciation"
])

with tab1:
    # Pronunciation Quiz Tile (full width)
    quiz_tile(speech_rate=rate_slider)

with tab2:
    # Spelling Checker Tile (full width)
    spelling_checker_tile(speech_rate=rate_slider)

with tab3:
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">📄 PDF Word Pronunciation</div>', unsafe_allow_html=True)
        st.caption("💡 Ask a parent or teacher for help picking the file if you're not sure!")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_file_uploader")
        pdf_words = []
        if pdf_file is not None:
            with st.spinner("📖 Reading your PDF..."):
                text = read_pdf_text(pdf_file)
                pdf_words = extract_words_from_text(text)
            st.write(f"Extracted {len(pdf_words)} unique words from PDF.")
            selected_word = st.selectbox("Select a word to learn pronunciation:", pdf_words)
            phones = pronouncing.phones_for_word(selected_word)
            if phones:
                st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                if st.button("Speak PDF Word", key="pdf_speak_btn"):
                    play_audio(selected_word, rate=rate_slider)
            else:
                st.warning("Pronunciation not found for this word.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">⌨️ Manual Word Pronunciation</div>', unsafe_allow_html=True)
        word = st.text_input("Enter a word:")
        if word:
            phones = pronouncing.phones_for_word(word.lower())
            if phones:
                st.markdown(f'<span class="pronunciation">Pronunciation (ARPAbet): {phones[0]}</span>', unsafe_allow_html=True)
                if st.button("Speak", key="manual_speak_btn"):
                    play_audio(word, rate=rate_slider)
            else:
                st.warning("Pronunciation not found for this word.")
        st.markdown('</div>', unsafe_allow_html=True)
