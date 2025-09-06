import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import requests
import time

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; padding: 20px; }
    .title { color: #1f77b4; font-size: 36px; text-align: center; font-weight: bold; }
    .subtitle { color: #4b5e6d; font-size: 18px; text-align: center; margin-bottom: 20px; }
    .result-card { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .result-title { color: #2c3e50; font-size: 20px; font-weight: bold; }
    .result-text { color: #34495e; font-size: 16px; }
    .footer { text-align: center; color: #7f8c8d; font-size: 14px; margin-top: 20px; }
    .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
    .stTextArea textarea { border-radius: 5px; border: 1px solid #d1d5db; }
    </style>
""", unsafe_allow_html=True)

# Initialize BERT model
try:
    model = BertForSequenceClassification.from_pretrained('bert_model')
    tokenizer = BertTokenizer.from_pretrained('bert_model')
except Exception as e:
    st.error(f"Error loading BERT model: {str(e)}")
    model, tokenizer = None, None

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# BERT prediction
def predict_bert(text):
    if not model or not tokenizer:
        return 'BERT model not loaded', 0.0
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return 'No valid text provided', 0.0
    inputs = tokenizer(cleaned, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=1).item()
    return 'Real' if label == 1 else 'Fake', probs[0][label].item()

# Web verification with Google Custom Search API
def verify_web(text):
    try:
        api_key = "Enter your api key"  
        cx = "enter ypur SE key"  
        
        # Construct search query
        query = text[:100].replace(' ', '+').replace('’', '%27')
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&num=5"
        time.sleep(1)  # Minimal delay for API
        
        # Fetch results
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract results
        items = data.get('items', [])
        snippet_texts = [item.get('snippet', '') for item in items]
        urls = [item.get('link', '') for item in items]
        
        # Credible domains
        credible_domains = [
            'bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com', 'theguardian.com',
            'timesofindia.indiatimes.com', 'ndtv.com', 'hindustantimes.com', 'indianexpress.com'
        ]
        
        # Check for credible sources
        credible_sources = [url for url in urls if any(domain in url for domain in credible_domains)]
        
        if credible_sources:
            return 'Real', 0.9, f"Found in credible sources: {', '.join(credible_sources[:2])}. Snippets: {' '.join(snippet_texts[:2])}"
        elif snippet_texts:
            return 'Uncertain', 0.5, f"No credible sources found. Snippets: {' '.join(snippet_texts[:2])}"
        else:
            return 'Fake', 0.2, "No relevant sources found."
    except Exception as e:
        return 'Uncertain', 0.5, f"Web verification failed. Try again later. Error: {str(e)}"

# Streamlit UI
st.markdown('<div class="title">Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter news text to check if it\'s real or fake using BERT and web verification.</div>', unsafe_allow_html=True)

# Example headlines
with st.expander("See Example Headlines"):
    st.write("""
    - **Real**: India’s parliament passed the Promotion and Regulation of Online Gaming Bill, 2025, banning real-money online games to combat addiction and financial risks.
    - **Real**: Dream11 halted real-money contests after the 2025 online gaming ban.
    - **Fake**: India’s 2025 gaming ban was secretly funded by foreign casinos to eliminate local competition.
    """)

# Input area
user_input = st.text_area('News Text:', value="Dream11 halted real-money contests after the 2025 online gaming ban.", height=100)

# Predict button with loading spinner
if st.button('Predict'):
    if user_input:
        with st.spinner('Analyzing news text...'):
            # BERT prediction
            bert_result, bert_confidence = predict_bert(user_input)
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">BERT Prediction</div>
                    <div class="result-text">Result: <b>{bert_result}</b> (Confidence: {bert_confidence:.2f})</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Web verification
            web_result, web_confidence, web_explanation = verify_web(user_input)
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Web Verification</div>
                    <div class="result-text">Result: <b>{web_result}</b> (Confidence: {web_confidence:.2f})</div>
                    <div class="result-text">Explanation: {web_explanation}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # BERT overconfidence warning
            if bert_result == 'Real' and bert_confidence > 0.95 and web_result in ['Fake', 'Uncertain']:
                st.warning("BERT may be overconfident. Web verification suggests this news may not be reliable.")
    else:
        st.error('Please enter news text.')

# Footer
st.markdown('<div class="footer">Built with Streamlit by Himanshi | Fake News Detection Project, 2025</div>', unsafe_allow_html=True)

