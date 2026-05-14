import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources once."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return set()

stop_words_english = download_nltk_resources()

def preprocess_text(text: str) -> str:
    """Clean and preprocess the input text for prediction."""
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    clean_tokens = [t for t in tokens if t not in stop_words_english and len(t) > 1]
    
    return " ".join(clean_tokens)
