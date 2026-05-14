import streamlit as st
import joblib
from modules.preprocessing import preprocess_text

@st.cache_resource
def load_models():
    """Load machine learning models and components."""
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        chi2_selector = joblib.load('chi2_selector.pkl')
        dt_model = joblib.load('decision_tree_model.pkl')
        return vectorizer, chi2_selector, dt_model
    except Exception as e:
        st.error(f"Error loading models. Ensure '.pkl' files exist in the directory. Details: {e}")
        return None, None, None

def predict_news(text: str, vectorizer, chi2_selector, dt_model) -> str:
    """Predict the category of the given news text."""
    if not vectorizer or not chi2_selector or not dt_model:
        return "Model not loaded properly"
        
    # Preprocess text
    clean_text = preprocess_text(text)
    if not clean_text:
        return "Invalid or empty text"
        
    # Transform text
    text_tfidf = vectorizer.transform([clean_text])
    text_final = chi2_selector.transform(text_tfidf)
    
    # Predict
    prediction = dt_model.predict(text_final)
    return prediction[0]
