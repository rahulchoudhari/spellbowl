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
        
        # Welcome and name input section
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
                    Let's start your pronunciation journey
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form(key="name_form"):
                st.markdown("### 📝 Tell us about yourself")
                student_name_input = st.text_input(
                    "What's your name?",
                    placeholder="Enter your name here...",
                    key="student_name_input",
                    help="This helps us personalize your learning experience!"
                )
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    submit_name = st.form_submit_button("🚀 Start Learning", use_container_width=True)
                
                if submit_name:
                    if student_name_input and student_name_input.strip():
                        st.session_state.student_name = student_name_input.strip()
                        st.session_state.name_submitted = True
                        st.success(f"🎉 Welcome, {st.session_state.student_name}! Let's begin!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Please enter your name to continue.")
            
            st.markdown("---")
            st.info("💡 **Tip:** Enter your name above to start the quiz and track your progress!")
            return  # Stop here until name is submitted
        
        # Show personalized greeting after name is submitted
        col_greeting, col_change = st.columns([4, 1])
        with col_greeting:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1em; 
                        border-radius: 10px; 
                        margin-bottom: 1em;
                        border-left: 5px solid #667eea;'>
                <p style='margin: 0; color: #2a3b5d; font-size: 1.2em; font-weight: 600;'>
                    👤 Hello, <strong>{st.session_state.student_name}</strong>! Ready to ace some words? 🌟
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_change:
            if st.button("✏️ Change Name", key="change_name_btn", use_container_width=True):
                st.session_state.name_submitted = False
                st.session_state.student_name = ""
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
        word_source = st.radio(
            "Select word source:",
            options=["Predefined Source", "Upload PDF", "System Generated"],
            key="word_source_radio",
            horizontal=True
        )
        
        if word_source == "Predefined Source":
            # Get PDF files from datasource folder
            datasource_path = Path("datasource")
            quiz_pdf = None
            
            if datasource_path.exists() and datasource_path.is_dir():
                pdf_files = list(datasource_path.glob("*.pdf"))
                
                if pdf_files:
                    # Create a dropdown with available PDFs
                    pdf_names = [pdf.name for pdf in pdf_files]
                    selected_pdf_name = st.selectbox(
                        "Select a PDF from predefined sources:",
                        options=pdf_names,
                        key="predefined_pdf_select"
                    )
                    
                    # Load button for predefined PDF
                    if st.button("📥 Load Selected PDF", key="load_predefined_pdf_btn"):
                        selected_pdf_path = datasource_path / selected_pdf_name
                        
                        try:
                            reader = PyPDF2.PdfReader(str(selected_pdf_path))
                            text = ""
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + " "
                            
                            words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
                            # Preserve original case and sort alphabetically (case-insensitive sort)
                            unique_words = {word.lower(): word for word in words if word}
                            all_words = [unique_words[key] for key in sorted(unique_words.keys())]
                            
                            if not all_words:
                                st.error("No valid words found in PDF.")
                            else:
                                # Store all words and select first 50 by default
                                st.session_state.all_loaded_words = all_words
                                st.session_state.quiz_words = all_words[:50] if len(all_words) > 50 else all_words
                                st.session_state.word_source_type = "predefined"
                                
                                # Reset quiz state when new PDF is loaded
                                st.session_state.used_quiz_words = []
                                st.session_state.current_quiz_word = None
                                st.session_state.quiz_score = 0
                                st.session_state.quiz_total = 0
                                st.session_state.answer_submitted = False
                                st.session_state.wrong_attempts = []
                                st.session_state.quiz_history = []
                                st.session_state.last_pdf_name = selected_pdf_name
                                
                                st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words from {selected_pdf_name}!")
                                st.info("👇 Select word range below and click 'Get Random Word' to start!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error reading PDF: {str(e)}")
                else:
                    st.warning("📂 No PDF files found in 'datasource' folder.")
                    st.info("💡 Add PDF files to the 'datasource' folder to use predefined sources.")
            else:
                st.warning("📂 'datasource' folder not found.")
                st.info("💡 Create a 'datasource' folder and add PDF files to use predefined sources.")
                
        elif word_source == "Upload PDF":
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
                
                st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words from {difficulty_level}!")
                st.info("👇 Select word range below and click 'Get Random Word' to start!")
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
                # Preserve original case and sort alphabetically (case-insensitive sort)
                unique_words = {word.lower(): word for word in words if word}
                all_words = [unique_words[key] for key in sorted(unique_words.keys())]
                
                if not all_words:
                    st.error("No valid words found in PDF. Please upload a different PDF.")
                    return
                
                # Store all words and select first 50 by default
                st.session_state.all_loaded_words = all_words
                st.session_state.quiz_words = all_words[:50] if len(all_words) > 50 else all_words
                st.session_state.word_source_type = "pdf"
                
                # Reset quiz state when new PDF is loaded
                st.session_state.used_quiz_words = []
                st.session_state.current_quiz_word = None
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.answer_submitted = False
                st.session_state.wrong_attempts = []
                st.session_state.quiz_history = []
                st.session_state.last_pdf_name = quiz_pdf.name
                
                st.success(f"✅ Loaded {len(st.session_state.all_loaded_words)} words from PDF!")
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
                            st.success(f"✅ Selected words {start_range} to {end_range} ({end_range - start_range + 1} words)")
                            st.rerun()
                        else:
                            st.error("Start word must be less than or equal to end word!")
                
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
                                # Debug: Show that word was selected
                                st.toast(f"Word selected! Click 'Play Pronunciation' to hear it.", icon="✅")
                                st.rerun()
                        else:
                            st.button("🎲 Get Random Word", key="random_word_btn", disabled=True)
                            st.caption("⚠️ Answer the current word first or skip it")
                    else:
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
                                Final Score: {st.session_state.quiz_score}/{st.session_state.quiz_total}
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

