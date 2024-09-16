import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from anthropic import Anthropic
import faiss
from pyvis.network import Network
import streamlit.components.v1 as components

# Initialize Claude client
def init_anthropic_client():
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

# Load and clean data
def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')
    
    data.columns = data.columns.str.strip()
    return data

# Call Claude for analysis
def call_claude(messages):
    try:
        system_message = messages[0]['content']
        user_message = messages[1]['content']
        
        response = client.completions.create(
            model="claude-2.1",
            prompt=f"{system_message}\n\nHuman: {user_message}\n\nAssistant:",
            max_tokens_to_sample=1500,
            temperature=0.7
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

# Create FAISS index for vector search
def create_vector_index(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Matter Description'].fillna(''))
    
    faiss_index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
    faiss_index.add(np.array(tfidf_matrix.toarray(), dtype=np.float32))
    
    return faiss_index, tfidf

# Extract conflict info from data
def extract_conflict_info(data, client_name=None, client_email=None, client_phone=None, faiss_index=None, tfidf=None):
    # Ensure case-insensitive comparison and handle empty values
    if client_name:
        client_name = client_name.lower()
    
    # Check for direct match in the 'Client Name'
