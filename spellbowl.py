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
from pathlib import Path
from nltk.corpus import wordnet
from pkg_resources import resource_stream

# Page Configuration
st.set_page_config(
    page_title="SpellBowl - Master Pronunciation & Spelling",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def play_audio(text, rate=100):
    """Play text using Google TTS with specified speech rate."""
    tts = gTTS(text=text, lang='en', slow=(rate < 80))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_file = fp.name
        tts.save(temp_file)
        
        # Display audio player with autoplay enabled
        with open(temp_file, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3', autoplay=True)
        
        # Also try system playback as fallback
        try:
            os.system(f"ffplay -nodisp -autoexit {temp_file} >/dev/null 2>&1 &")
        except Exception:
            pass
    
    # Schedule cleanup after a delay (async-like)
    try:
        import threading
        threading.Timer(10.0, lambda: Path(temp_file).unlink(missing_ok=True)).start()
    except Exception:
        pass

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

def get_system_generated_words(level):
    """Get system generated words based on difficulty level."""
    # Level 1: School Grade 1-3 (Simple 3-5 letter words)
    level1_words = [
        'cat', 'dog', 'run', 'jump', 'play', 'sun', 'moon', 'star', 'tree', 'bird',
        'fish', 'book', 'ball', 'home', 'door', 'window', 'table', 'chair', 'water', 'food',
        'apple', 'happy', 'good', 'make', 'work', 'help', 'give', 'take', 'open', 'close',
        'walk', 'talk', 'read', 'write', 'smile', 'laugh', 'sleep', 'wake', 'eat', 'drink',
        'love', 'like', 'want', 'need', 'have', 'look', 'see', 'hear', 'feel', 'know'
    ]
    
    # Level 2: School Grade 4-6 (Medium 5-7 letter words)
    level2_words = [
        'school', 'teacher', 'student', 'pencil', 'science', 'history', 'culture', 'nature', 'animal', 'garden',
        'computer', 'library', 'picture', 'question', 'answer', 'problem', 'solution', 'practice', 'example', 'exercise',
        'morning', 'evening', 'afternoon', 'weather', 'summer', 'winter', 'spring', 'autumn', 'mountain', 'river',
        'ocean', 'island', 'forest', 'desert', 'village', 'city', 'country', 'planet', 'universe', 'galaxy',
        'friend', 'family', 'parent', 'brother', 'sister', 'teacher', 'doctor', 'engineer', 'artist', 'musician'
    ]
    
    # Level 3: School Grade 7-10 (Advanced 7-10 letter words)
    level3_words = [
        'knowledge', 'education', 'experience', 'technology', 'information', 'communication', 'environment', 'population', 'government', 'democracy',
        'literature', 'mathematics', 'geography', 'biology', 'chemistry', 'physics', 'philosophy', 'psychology', 'sociology', 'economics',
        'organization', 'development', 'achievement', 'opportunity', 'responsibility', 'independence', 'confidence', 'intelligence', 'creativity', 'imagination',
        'community', 'society', 'tradition', 'ceremony', 'celebration', 'generation', 'revolution', 'evolution', 'discovery', 'invention',
        'beautiful', 'wonderful', 'excellent', 'important', 'necessary', 'different', 'particular', 'separate', 'complete', 'continue'
    ]
    
    # Level 4: School Grade 10-12 (Complex 8+ letter words)
    level4_words = [
        'architecture', 'extraordinary', 'sophisticated', 'comprehensive', 'contemporary', 'fundamental', 'revolutionary', 'unprecedented', 'characteristic', 'philosophical',
        'psychological', 'technological', 'environmental', 'international', 'constitutional', 'mathematical', 'archaeological', 'astronomical', 'biological', 'geographical',
        'responsibility', 'entrepreneurship', 'sustainability', 'globalization', 'infrastructure', 'transportation', 'communication', 'investigation', 'authentication', 'determination',
        'accomplishment', 'establishment', 'implementation', 'transformation', 'interpretation', 'representation', 'pronunciation', 'participation', 'collaboration', 'demonstration',
        'magnificent', 'extraordinary', 'intellectual', 'controversial', 'enthusiastic', 'spontaneous', 'simultaneous', 'anonymous', 'autonomous', 'synonymous'
    ]
    
    level_map = {
        'Level 1 (Grade 1-3)': level1_words,
        'Level 2 (Grade 4-6)': level2_words,
        'Level 3 (Grade 7-10)': level3_words,
        'Level 4 (Grade 10-12)': level4_words
    }
    
    return level_map.get(level, level1_words)

def quiz_tile(speech_rate=100):
    """Interactive pronunciation quiz tile."""
    with st.container():
        st.markdown('<div class="tile"><div class="tile-title">🎯 Pronunciation Quiz</div>', unsafe_allow_html=True)
        
        # Initialize session state
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
        
        # Word source selection
        st.markdown("### 📚 Choose Word Source")
        word_source = st.radio(
            "Select word source:",
            options=["Upload PDF", "System Generated"],
            key="word_source_radio",
            horizontal=True
        )
        
        if word_source == "Upload PDF":
            # PDF uploader for quiz words
            quiz_pdf = st.file_uploader("Upload PDF for quiz words", type=["pdf"], key="quiz_pdf_uploader")
        else:
            quiz_pdf = None
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
                st.session_state.quiz_words = random.sample(system_words, min(50, len(system_words)))
                
                # Reset quiz state when new words are loaded
                st.session_state.used_quiz_words = []
                st.session_state.current_quiz_word = None
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.answer_submitted = False
                st.session_state.wrong_attempts = []
                st.session_state.quiz_history = []
                st.session_state.last_pdf_name = None  # Clear PDF tracking
                
                st.success(f"✅ Loaded {len(st.session_state.quiz_words)} words from {difficulty_level}!")
                st.info("👇 Click 'Get Random Word' below to start!")
                st.rerun()
        
        # Check if PDF was just uploaded and needs processing
        if quiz_pdf is not None and (not st.session_state.quiz_words or 'last_pdf_name' not in st.session_state or st.session_state.get('last_pdf_name') != quiz_pdf.name):
            try:
                reader = PyPDF2.PdfReader(quiz_pdf)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Words with 4+ letters
                all_words = sorted(set(word.lower() for word in words if word))
                
                if not all_words:
                    st.error("No valid words found in PDF. Please upload a different PDF.")
                    return
                
                # Select up to 50 random words
                import random
                if len(all_words) > 50:
                    st.session_state.quiz_words = random.sample(all_words, 50)
                else:
                    st.session_state.quiz_words = all_words
                
                # Reset quiz state when new PDF is loaded
                st.session_state.used_quiz_words = []
                st.session_state.current_quiz_word = None
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.answer_submitted = False
                st.session_state.wrong_attempts = []
                st.session_state.quiz_history = []
                st.session_state.last_pdf_name = quiz_pdf.name
                
                st.success(f"✅ Loaded {len(st.session_state.quiz_words)} words for quiz!")
                st.info("👇 Click 'Get Random Word' below to start!")
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return
        
        if st.session_state.quiz_words:
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
                
                # Create visual progress bar with emojis
                history_display = ""
                for idx, result in enumerate(st.session_state.quiz_history, 1):
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
                </div>
                """, unsafe_allow_html=True)
                
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
                st.info("👉 Click 'Get Random Word' to start!")
            elif not st.session_state.answer_submitted:
                st.info("🎧 Listen and type your answer!")
            
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
                                # Debug: Show that word was selected
                                st.toast(f"Word selected! Click 'Play Pronunciation' to hear it.", icon="✅")
                                st.rerun()
                        else:
                            st.button("🎲 Get Random Word", key="random_word_btn", disabled=True)
                            st.caption("⚠️ Answer the current word first or skip it")
                    else:
                        st.info("🎉 Quiz Complete!")
                        if st.button("🔄 Reset Quiz", key="reset_quiz_btn"):
                            st.session_state.used_quiz_words = []
                            st.session_state.current_quiz_word = None
                            st.session_state.quiz_score = 0
                            st.session_state.quiz_total = 0
                            st.session_state.answer_submitted = False
                            st.session_state.quiz_history = []
                            st.rerun()
                else:
                    st.warning("Please upload a PDF first to load quiz words.")
            
            with col_b:
                if st.session_state.current_quiz_word:
                    if st.button("🔊 Play Pronunciation", key="quiz_play_btn"):
                        play_audio(st.session_state.current_quiz_word, rate=speech_rate)
                else:
                    st.button("🔊 Play Pronunciation", key="quiz_play_btn", disabled=True)
            
            with col_c:
                if st.session_state.current_quiz_word:
                    if st.button("⏭️ Skip Word", key="skip_word_btn"):
                        if not st.session_state.answer_submitted:
                            st.session_state.used_quiz_words.append(st.session_state.current_quiz_word)
                            st.session_state.current_quiz_word = None
                            st.rerun()
                else:
                    st.button("⏭️ Skip Word", key="skip_word_btn", disabled=True)
            
            # Show the quiz interface when a word is selected
            if st.session_state.current_quiz_word:
                st.markdown("---")
                st.info("🎧 Listen to the pronunciation and spell the word below:")
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
                        
                        if user_answer.lower() == correct_word:
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
                            similarity = difflib.SequenceMatcher(None, user_answer.lower(), correct_word).ratio()
                            st.session_state.quiz_history.append(0)  # Track wrong answer
                            
                            # Track wrong attempt for revision
                            st.session_state.wrong_attempts.append({
                                'correct': correct_word,
                                'your_answer': user_answer,
                                'similarity': similarity * 100
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
                                <p style='margin: 0; font-size: 3em;'>❌</p>
                                <p style='margin: 0.5em 0 0 0; color: #991b1b; font-size: 1.5em; font-weight: 800;'>
                                    NOT QUITE!
                                </p>
                                <p style='margin: 0.3em 0 0 0; color: #b91c1c; font-size: 1em;'>
                                    You spelled: <strong>{user_answer}</strong>
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
                            
                            # Show correct spelling prominently
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
                                    {correct_word.upper()}
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
                            
                            # Give hints based on similarity
                            if similarity > 0.7:
                                st.info("💡 You're very close! Just a few letters off.")
                            elif similarity > 0.5:
                                st.info(f"💡 The word has {len(correct_word)} letters and you got most of them right.")
                            else:
                                st.info(f"💡 Tip: The word starts with **'{correct_word[0].upper()}'** and has **{len(correct_word)} letters**.")
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
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                                    padding: 1em; 
                                    border-radius: 10px; 
                                    border-left: 4px solid #f59e0b; 
                                    margin: 0.8em 0;'>
                            <p style='margin: 0; color: #92400e; font-size: 0.85em; font-weight: 600;'>
                                #{idx} - Similarity: {attempt['similarity']:.1f}%
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #b45309;'>
                                <strong>Your Answer:</strong> <span style='text-decoration: line-through;'>{attempt['your_answer']}</span>
                            </p>
                            <p style='margin: 0.3em 0 0 0; color: #065f46; font-weight: 700;'>
                                <strong>Correct:</strong> {attempt['correct'].upper()}
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
        else:
            st.info("📤 Upload a PDF to start the pronunciation quiz!")
            st.markdown("""
            **How it works:**
            1. Upload a PDF document
            2. System will randomly select 50 words
            3. Click 'Get Random Word' to start
            4. Listen to pronunciation (no spelling shown!)
            5. Type what you heard and check your answer
            """)
        
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

rate_slider = st.slider("Adjust speech rate (%)", min_value=30, max_value=150, value=100, step=10, key="speech_rate_slider")

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
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_file_uploader")
        pdf_words = []
        if pdf_file is not None:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            import re
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            pdf_words = sorted(set(word.lower() for word in words))
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

