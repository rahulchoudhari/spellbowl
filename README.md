# 🎯 SpellBowl - Master Pronunciation & Spelling

A comprehensive pronunciation and spelling learning application built with Streamlit. Perfect for students, teachers, and anyone looking to improve their spelling and pronunciation skills!

## ✨ Features

### 🔐 User Authentication System
- **Secure Login & Registration:**
  - Username and password-based authentication
  - New user registration with validation
  - Minimum 3 characters for username
  - Minimum 4 characters for password
  - ## 📊 Data Files

The application creates and maintains two JSON files:

### users.json
Stores user account information:
```json
{
  "username": {
    "password": "user_password",
    "full_name": "User Full Name",
    "created_at": "2025-10-18 00:00:00",
    "total_quizzes": 5,
    "best_score": 100.0
  }
}
```

### leaderboard.json
Stores cumulative quiz statistics per user:
```json
{
  "username": {
    "name": "Full Name",
    "total_score": 50,
    "total_questions": 75,
    "total_quizzes": 5,
    "best_accuracy": 100.0,
    "avg_accuracy": 85.3,
    "last_quiz_date": "2025-10-18 00:19:14",
    "quiz_history": [
      {
        "score": 10,
        "total": 15,
        "accuracy": 66.7,
        "timestamp": "2025-10-18 00:15:00",
        "word_source": "predefined"
      }
    ]
  }
}
```

**Note:** 
- Backup these files regularly to preserve user data
- In production, passwords should be hashed (not stored in plain text)
- Files are automatically created on first use

## 📊 Troubleshooting

### Login/Authentication Issues
- **Forgot Password**: No recovery mechanism yet (admin must edit users.json)
- **Username Taken**: Choose a different username
- **Login Failed**: Check username and password (case-sensitive)
- **Session Lost**: Re-login if session expires

### Leaderboard Issues
- **Stats Not Updating**: Refresh the page after quiz completion
- **Missing User**: Ensure you completed at least one quiz while logged in
- **Duplicate Entries (Old)**: System auto-migrates old format to new

### AxiosError: Network Error (Samsung Tablets/Mobile Devices)sword confirmation during registration
  - Unique username enforcement
  
- **User Profile Management:**
  - Personalized greetings with full name and username
  - Real-time stats display (total quizzes, best score)
  - Logout functionality
  - Session persistence

### 🏆 Comprehensive Leaderboard System
- **Cumulative User Tracking:**
  - Single entry per user (no duplicates)
  - Automatic score aggregation across quizzes
  - Best accuracy and average accuracy tracking
  - Total quizzes, total score, and total questions tracked
  - Complete quiz history for each user
  
- **Sidebar Leaderboard:**
  - Always visible top 5 performers
  - Shows best accuracy, average accuracy, and total quizzes
  - Medal indicators (🥇🥈🥉) for top 3
  - Auto-updates after each quiz
  
- **Detailed Leaderboard Tabs:**
  - **Top Performers**: Ranked by best accuracy
  - **My Scores**: Personal stats and complete quiz history
  - **All Users**: Overview of all registered users

### 🎯 Pronunciation Quiz (Main Feature)
- **User-Based Learning Experience:**
  - Login required to track progress
  - Personalized dashboard with stats
  - Individual quiz history tracking
  - Best score and average accuracy calculation
  
- **Three Word Sources:**
  - 📚 **Predefined Source**: 750 curated words with proper formatting (multi-word phrases, hyphens, apostrophes)
  - 📄 **Upload PDF**: Extract words from any PDF document
  - 🎓 **System Generated**: Pre-curated word lists by difficulty level
  
- **Four Difficulty Levels (System Generated):**
  - Level 1 (Grade 1-3): Simple 3-5 letter words
  - Level 2 (Grade 4-6): Medium 5-7 letter words
  - Level 3 (Grade 7-10): Advanced 7-10 letter words
  - Level 4 (Grade 10-12): Complex 10+ letter words
  
- **Flexible Word Range Selection:**
  - Preset ranges: 1-10, 11-25, 26-50, 51-100, or All Words
  - Custom range selector: Choose any start and end word number
  - Visual indicator showing current practice range
  
- **Competition Mode:**
  - Enable timed challenges for each question
  - Adjustable timer: 5-120 seconds per question
  - Timer starts after audio plays
  - Color-coded countdown (green → orange → red)
  - Time expiration handling with feedback
  
- **Interactive Quiz System:**
  - Listen to word pronunciation (no spelling shown)
  - Type what you hear with case-sensitive checking
  - Press Enter to submit answers quickly
  - Get instant feedback with detailed error types:
    - ✅ Correct spelling
    - 🔡 Case sensitivity errors (spelling correct, capitalization wrong)
    - ❌ Spelling errors
  - Similarity score percentage for wrong answers
  - Contextual hints based on error type
  
- **Performance Tracking:**
  - Real-time score display (correct/total)
  - Accuracy percentage
  - Remaining words counter
  - Visual performance tracker with emoji indicators
  - Hover over emojis to see which word it was
  - Detailed performance history
  - Current streak and best streak tracking
  
- **Revision List:**
  - Track all wrong attempts with error type badges
  - Color-coded by error type (timeout, case, spelling)
  - Shows similarity scores
  - Practice pronunciation of missed words
  - Clear revision list when done
  
- **Smart Word Handling:**
  - Preserves multi-word phrases: "time zone", "Rio Grande"
  - Handles hyphenated words: "tip-in", "about-face"
  - Handles apostrophes: "you're", "baker's dozen"
  - Maintains proper capitalization: "Big Dipper", "British Columbia"
  - Case-sensitive answer checking
  - Alphabetical sorting with case preservation

### 📝 Spelling Checker & Pronunciation Helper
- Check spelling of any word
- Get suggestions for misspelled words
- Similarity percentage for corrections
- ARPAbet pronunciation display
- Audio pronunciation playback
- Word hints with definitions, synonyms, and antonyms

### 📄 PDF Word Pronunciation
- Upload any PDF document
- Extract and list all unique words
- Universal extraction supports various PDF formats
- Handles text-based PDFs (not scanned images)
- Select words to learn pronunciation
- ARPAbet notation display
- Adjustable speech rate

### ✍️ Manual Word Pronunciation
- Enter any word manually
- Instant pronunciation lookup
- ARPAbet phonetic notation
- Audio playback with adjustable speed

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download this repository:**
   ```bash
   cd spellbowl
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run spellbowl.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📦 Dependencies

- **streamlit** (>=1.28.0): Web framework for the app
- **PyPDF2** (>=3.0.0): PDF text extraction
- **gTTS** (>=2.4.0): Google Text-to-Speech engine
- **pronouncing** (>=0.2.0): ARPAbet pronunciation lookup
- **nltk** (>=3.8.1): Natural Language Toolkit for spell checking

## 🎮 How to Use

### First Time Setup

1. **Register an Account:**
   - Click the "📝 Register" button
   - Enter your full name
   - Choose a unique username (min 3 characters)
   - Create a password (min 4 characters)
   - Confirm your password
   - Click "✨ Create Account"

2. **Login:**
   - Click the "🔑 Login" button (default view)
   - Enter your username and password
   - Click "🚀 Login"
   - Welcome back message displays your stats

### Pronunciation Quiz

1. **After Login:**
   - See your personalized dashboard with username and stats
   - View sidebar leaderboard showing top 5 performers
   - Logout button available in top right

2. **Choose Competition Mode (Optional):**
   - Check "Enable Competition Mode" for timed challenges
   - Adjust timer (5-120 seconds per question)
   - Timer starts after audio plays

3. **Choose Word Source:**
   - **Predefined Source**: 750 curated words ready to use
   - **Upload PDF**: Use your own document
   - **System Generated**: Choose difficulty level (1-4)

4. **Load Words:**
   - Click "Load Predefined Words" for curated list
   - Or upload PDF and it auto-loads
   - Or select difficulty and click "Load System Words"

5. **Select Word Range:**
   - Use preset buttons: 1-10, 11-25, 26-50, 51-100, or All Words
   - Or use custom range: Enter start and end word numbers
   - Click "Apply" for custom ranges

6. **Start Quiz:**
   - Click "Get Random Word"
   - Click "Play Pronunciation" to hear the word
   - Timer starts (if competition mode enabled)
   - Type what you heard (case-sensitive!)
   - Press Enter or click "Check Answer"

7. **Review Feedback:**
   - ✅ Correct: Celebration with balloons!
   - 🔡 Case Error: Shows correct capitalization
   - ❌ Spelling Error: Shows similarity score and hints
   - ⏰ Time Expired: Shows correct answer
   - Click "Next Word" to continue

8. **Track Progress:**
   - View score, accuracy, and remaining words
   - See performance tracker with emoji history
   - Monitor current streak and best streak
   - Review wrong attempts in revision list

9. **Practice Mistakes:**
   - Expand "View All Wrong Attempts"
   - Click "Hear" to practice pronunciation
   - See error types and similarity scores
   - Clear list when done

### Spelling Checker

1. Navigate to the "Spelling Checker & Pronunciation Helper" tab
2. Enter any word in the text field
3. Get instant feedback on spelling
4. Click "Get Hint" for definitions, synonyms, and antonyms
5. See pronunciation and play audio

### Adjusting Speech Rate

- Use the slider at the top: 30% (slow) to 150% (fast)
- Perfect for different learning speeds
- Applies to all pronunciation features

## 🏫 Use Cases

### For Teachers
- Create student accounts and track individual progress
- Use 750 predefined curated words or create custom tests from curriculum PDFs
- Enable competition mode for timed spelling bees
- Choose appropriate difficulty levels for different grades
- Monitor class performance through leaderboard
- View detailed statistics for each student
- Track best scores and average accuracy
- Monitor common mistakes through revision lists
- Use as a classroom tool for pronunciation practice
- Create custom word ranges for targeted learning
- Review complete quiz history for assessment

### For Students
- Create personal account to track progress over time
- Practice spelling independently with instant feedback
- Use competition mode to challenge yourself
- View your ranking on the leaderboard
- Track personal best and average scores
- Review difficult words from textbooks via PDF upload
- Prepare for spelling bees with timed practice
- Build vocabulary with pronunciation and definitions
- See complete quiz history and improvement trends
- Learn case-sensitive spelling (important for proper nouns)
- Get detailed feedback on error types (case vs spelling)
- Compete with classmates on the leaderboard

### For Parents
- Create accounts for children to monitor progress
- Track improvement through cumulative statistics
- Help children with homework using predefined word lists
- Create fun spelling challenges with timer
- View best scores and quiz history
- Compare performance with peers (leaderboard)
- Supplement school education with targeted practice
- Practice at appropriate difficulty levels
- Use custom word ranges for focused learning

## 🌐 Deployment

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy in one click
5. Share the URL with students

### Self-Hosted
- Run on any server with Python
- Supports multiple concurrent users
- Each user has isolated session state
- Perfect for classroom environments

## 🔧 Technical Details

### Authentication & User Management
- **User Authentication**:
  - Username/password based login system
  - Secure user registration with validation
  - Password storage (Note: In production, use hashed passwords!)
  - Session-based authentication
  - Logout functionality
  
- **Data Storage**:
  - `users.json`: Stores user credentials and basic info
  - `leaderboard.json`: Stores cumulative quiz statistics
  - JSON format for easy backup and portability
  - Auto-migration from old format to new format

### Leaderboard System
- **Cumulative Tracking**:
  - Single entry per user (no duplicates)
  - Automatic aggregation of quiz results
  - Best accuracy tracking
  - Average accuracy calculation
  - Total score across all quizzes
  - Total questions attempted
  - Complete quiz history array
  
- **Data Structure**:
  ```json
  {
    "username": {
      "name": "Full Name",
      "total_score": 50,
      "total_questions": 75,
      "total_quizzes": 5,
      "best_accuracy": 100.0,
      "avg_accuracy": 85.3,
      "last_quiz_date": "2025-10-18 00:19:14",
      "quiz_history": [...]
    }
  }
  ```

### Session Isolation & Personalization
- Each logged-in user has isolated session state
- Individual quiz sessions with secure authentication
- Scores and progress linked to user accounts
- Safe for multi-user environments
- Real-time stats display
- Persistent user data across sessions

### Smart Word Extraction
- **Predefined Source**: 750 hand-curated words with:
  - Multi-word phrases: "time zone", "Big Dipper"
  - Hyphenated words: "tip-in", "about-face"
  - Apostrophe words: "you're", "baker's dozen"
  - Proper nouns: "Rio Grande", "Mount Rushmore"
  
- **PDF Upload**: Universal extraction algorithm:
  - Handles various PDF formats
  - Preserves multi-word phrases
  - Maintains case sensitivity
  - Removes duplicates while preserving original case
  - Alphabetical sorting (case-insensitive)

### Advanced Quiz Features
- **Case-Sensitive Checking**: Exact match required (important for proper nouns)
- **Error Type Detection**: Distinguishes between case errors and spelling errors
- **Similarity Scoring**: Uses SequenceMatcher for percentage match
- **Contextual Hints**: Different feedback based on error type and similarity
- **Timer System**: 
  - Starts after audio plays (prevents premature timing)
  - Color-coded countdown display
  - Auto-submit on timeout
  - Graceful timeout handling
  
### Performance Tracking
- **Real-time Metrics**: Score, accuracy, remaining words
- **Visual Progress**: Emoji-based history with hover tooltips
- **Streak Tracking**: Current streak and best streak
- **Detailed History**: Complete log of correct/wrong answers
- **Revision System**: Categorized by error type with color coding

### Performance Optimizations
- NLTK word list cached with `@st.cache_resource`
- WordNet data cached for hints
- Automatic cleanup of temporary audio files
- Efficient PDF processing with regex
- Responsive UI without blocking operations
- Session state management for smooth navigation

### Audio Processing
- Uses Google Text-to-Speech (gTTS)
- Automatic audio playback
- Temp file cleanup after 10 seconds
- Adjustable speech rate control

## 📊 Scalability

- **Development/Small Class**: 1-10 concurrent users
- **Streamlit Cloud**: 50-100 concurrent users
- **Self-Hosted**: Hundreds of users with proper infrastructure
- **Enterprise**: Unlimited with load balancing

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share feedback

## 📝 License

This project is open source and available for educational purposes.

## 🎓 Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [gTTS](https://github.com/pndurette/gTTS) - Google Text-to-Speech
- [NLTK](https://www.nltk.org/) - Natural Language Processing
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF processing
- [Pronouncing](https://pronouncing.readthedocs.io/) - Phonetic lookup

## 📞 Support

For issues or questions:
1. Check the [IMPROVEMENTS.md](IMPROVEMENTS.md) file for troubleshooting
2. Review this README for usage guidance
3. Test with sample PDFs to verify functionality

## 🎯 Quick Start Example

```bash
# Install and run in 3 commands
pip install -r requirements.txt
streamlit run spellbowl.py
# App opens at http://localhost:8501
```

## 🆕 Latest Updates

### Version 3.0 Features (New! 🎉)
- ✅ **User Authentication System**: Secure login and registration
- ✅ **Cumulative Leaderboard**: Single entry per user with aggregated stats
- ✅ **Sidebar Leaderboard**: Always-visible top 5 performers
- ✅ **Personal Dashboard**: Real-time stats (total quizzes, best score, average)
- ✅ **Quiz History Tracking**: Complete history of all quiz attempts
- ✅ **Best & Average Accuracy**: Track improvement over time
- ✅ **User Profile Management**: Logout functionality and session persistence
- ✅ **Automatic Data Migration**: Seamlessly converts old format to new

### Version 2.0 Features
- ✅ **Personalized Experience**: Name-based progress tracking
- ✅ **Competition Mode**: Timed challenges with adjustable timer
- ✅ **750 Predefined Words**: Curated word list ready to use
- ✅ **Flexible Word Ranges**: Preset and custom range selection
- ✅ **Case-Sensitive Checking**: Important for proper nouns
- ✅ **Error Type Detection**: Distinguishes case vs spelling errors
- ✅ **Performance Tracker**: Visual history with streaks
- ✅ **Smart Word Extraction**: Handles multi-word phrases, hyphens, apostrophes
- ✅ **Enhanced Feedback**: Contextual hints based on error type
- ✅ **Revision System**: Categorized by error type with color coding
- ✅ **Word Hints**: Definitions, synonyms, and antonyms
- ✅ **Universal PDF Support**: Works with various PDF formats

### Key Improvements
- User accounts with secure authentication
- Cumulative statistics per user (no duplicate entries)
- Best score and average accuracy tracking
- Complete quiz history for each user
- Sidebar leaderboard with top performers
- Real-time stats updates after each quiz
- Timer starts after audio plays (prevents premature timing)
- Case-sensitive answer validation
- Detailed error feedback (case vs spelling)
- Multi-word phrase support ("time zone", "Rio Grande")
- Hyphenated word support ("tip-in", "about-face")
- Apostrophe word support ("you're", "baker's dozen")
- Visual performance tracking with emoji indicators
- Streak tracking (current and best)
- Word position tracking in toasts
- Cleaner UI without debug information

## � Troubleshooting

### AxiosError: Network Error (Samsung Tablets/Mobile Devices)

If you encounter network errors on Samsung tablets or mobile devices:

**1. Browser Compatibility:**
- ✅ **Recommended:** Use Chrome or Firefox browser
- ❌ **Avoid:** Samsung Internet Browser (has WebSocket issues)

**2. Clear Browser Cache:**
```
Settings → Apps → Browser → Storage → Clear Cache
```

**3. Check Connection:**
- Ensure stable Wi-Fi connection
- Try disabling VPN if enabled
- Refresh the page (pull down to refresh)

**4. URL Access:**
- Use HTTPS: `https://mrespellbowl.streamlit.app`
- Avoid using bookmarks (they may cache old settings)
- Type URL directly or use a fresh link

**5. Mobile Settings:**
- Enable JavaScript
- Allow cookies for the site
- Disable battery optimization for browser
- Check if "Lite mode" or "Data saver" is disabled

**6. If Still Not Working:**
- Try on different device to confirm it's device-specific
- Contact us with:
  - Device model (e.g., Samsung Galaxy Tab A7)
  - Browser version
  - Error message screenshot

### Audio Not Playing
- Check device volume
- Tap the play button on audio player
- Some browsers require user interaction before autoplay
- Ensure internet connection (gTTS requires online access)

### PDF Upload Issues
- Ensure PDF is text-based (not scanned images)
- File size should be under 200MB
- Try a different PDF if extraction fails
- Check if PDF contains readable text (copy-paste test)
- Universal extraction works with most PDF formats

### Competition Mode Timer Issues
- Timer starts only after clicking "Play Pronunciation"
- Cannot submit answer before audio plays
- Timer stops automatically when answer is submitted
- Page refresh resets the timer

### Word Extraction Issues
- If fewer words than expected are extracted from PDF:
  - Check if PDF contains actual text (not images)
  - Verify words are at least 4 characters long
  - Some formatting might split multi-word phrases
  - Use predefined source for guaranteed 750 words

### Case Sensitivity
- Answers must match exact capitalization
- "california" ≠ "California" (case error)
- "Rio grande" ≠ "Rio Grande" (case error)
- Pay attention to proper nouns and capitalization

## �👥 Credits

**Made with ❤️ for MRE**

**Developers:**
- **Aashrita Choudhari** - Co-Creator & Developer
- **Rahul Choudhari** - Co-Creator & Developer

*Empowering students to master pronunciation and spelling, one word at a time.*

---

**Happy Learning! 📚✨**

Master spelling and pronunciation one word at a time with SpellBowl!
