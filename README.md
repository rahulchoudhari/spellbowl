# 🎯 SpellBowl - Master Pronunciation & Spelling

A comprehensive pronunciation and spelling learning application built with Streamlit. Perfect for students, teachers, and anyone looking to improve their spelling and pronunciation skills!

## ✨ Features

### 🎯 Pronunciation Quiz (Main Feature)
- **Two Word Sources:**
  - 📄 **Upload PDF**: Extract words from any PDF document
  - 🎓 **System Generated**: Pre-curated word lists by difficulty level
- **Four Difficulty Levels:**
  - Level 1 (Grade 1-3): Simple 3-5 letter words
  - Level 2 (Grade 4-6): Medium 5-7 letter words
  - Level 3 (Grade 7-10): Advanced 7-10 letter words
  - Level 4 (Grade 10-12): Complex 8+ letter words
- **Interactive Quiz System:**
  - Listen to word pronunciation (no spelling shown)
  - Type what you hear
  - Get instant feedback with similarity score
  - Track your score and accuracy
  - Press Enter to submit answers
- **Revision List:**
  - Track all wrong attempts
  - Review mistakes with similarity scores
  - Practice pronunciation of missed words
  - Clear revision list when done

### 📝 Spelling Checker & Pronunciation Helper
- Check spelling of any word
- Get suggestions for misspelled words
- Similarity percentage for corrections
- ARPAbet pronunciation display
- Audio pronunciation playback

### 📄 PDF Word Pronunciation
- Upload any PDF document
- Extract and list all unique words
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

### Pronunciation Quiz

1. **Choose Word Source:**
   - Select "Upload PDF" to use your own document
   - Select "System Generated" for pre-built word lists

2. **For System Generated:**
   - Choose difficulty level (1-4)
   - Click "Load System Words"
   - 50 random words will be loaded

3. **For PDF Upload:**
   - Click "Browse files" and select a PDF
   - Up to 50 random words will be extracted automatically

4. **Start Quiz:**
   - Click "Get Random Word"
   - Click "Play Pronunciation" to hear the word
   - Type what you heard
   - Press Enter or click "Check Answer"
   - Get instant feedback with similarity score

5. **Review Mistakes:**
   - Expand "View All Wrong Attempts" to see revision list
   - Click "Hear" button to practice missed words
   - Track your progress with score and accuracy metrics

### Spelling Checker

1. Navigate to the "Spelling Checker & Pronunciation Helper" tab
2. Enter any word in the text field
3. Get instant feedback on spelling
4. See pronunciation and play audio

### Adjusting Speech Rate

- Use the slider at the top: 30% (slow) to 150% (fast)
- Perfect for different learning speeds
- Applies to all pronunciation features

## 🏫 Use Cases

### For Teachers
- Create custom spelling tests from curriculum PDFs
- Choose appropriate difficulty levels for different grades
- Track student progress and common mistakes
- Use as a classroom tool for pronunciation practice

### For Students
- Practice spelling independently
- Review difficult words from textbooks
- Prepare for spelling bees
- Build vocabulary with pronunciation
- Track improvement over time

### For Parents
- Help children with homework
- Create fun spelling challenges
- Monitor learning progress
- Supplement school education

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

### Session Isolation
- Each user gets their own quiz session
- Scores and progress are independent
- No data shared between users
- Safe for multi-user environments

### Performance Optimizations
- NLTK word list cached with `@st.cache_resource`
- Automatic cleanup of temporary audio files
- Efficient PDF processing
- Responsive UI without blocking operations

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

## 👥 Credits

**Made with ❤️ for MRE**

**Developers:**
- **Aashrita Choudhari** - Co-Creator & Developer
- **Rahul Choudhari** - Co-Creator & Developer

*Empowering students to master pronunciation and spelling, one word at a time.*

---

**Happy Learning! 📚✨**

Master spelling and pronunciation one word at a time with SpellBowl!
