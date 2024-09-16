import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from anthropic import Anthropic
import faiss

def init_anthropic_client():
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')
    
    data.columns = data.columns.str.strip()
    return data

def call_claude(messages):
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": messages[0]['content']},
                {"role": "user", "content": messages[1]['content']}
            ]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def create_vector_index(data):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data['Matter Description'].fillna(''))
        
        faiss_index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
        faiss_index.add(np.array(tfidf_matrix.toarray(), dtype=np.float32))
        
        return faiss_index, tfidf
    except Exception as e:
        st.error(f"Error creating vector index: {e}")
        return None, None

def extract_conflict_info(data, client_name, faiss_index, tfidf):
    try:
        query_vec = tfidf.transform([client_name]).toarray().astype(np.float32)
        _, I = faiss_index.search(query_vec, k=5)
        
        relevant_data = data.iloc[I[0]]

        if relevant_data.empty:
            return pd.DataFrame(columns=['Client', 'Conflict Type', 'Details'])

        messages = [
            {"role": "system", "content": "You are a legal assistant tasked with identifying potential conflicts of interest, opponents, and business owners related to a client."},
            {"role": "user", "content": f"""Analyze the following data for the client '{client_name}'. Identify:
    1. If the client has worked with the law firm before (indicating a potential conflict)
    2. Potential opponents of the client (look for 'vs' or similar indicators in the Matter or Matter Description)
    3. Any mentioned business owners related to the client

    For each identified item, provide the following in a structured format:
    - Client: The name of the client
    - Conflict Type: Either 'Prior Work', 'Potential Opponent', or 'Business Owner'
    - Details: Relevant details about the conflict, opponent, or business owner

    Here's the relevant data:

    {relevant_data.to_string()}

    Provide your analysis in a structured format that can be easily converted to a table."""}
        ]

        claude_response = call_claude(messages)
        if not claude_response:
            return pd.DataFrame(columns=['Client', 'Conflict Type', 'Details'])

        # Parse Claude's response into a structured format
        lines = claude_response.split('\n')
        parsed_data = []
        current_entry = {}
        for line in lines:
            if line.startswith('Client:'):
                if current_entry:
                    parsed_data.append(current_entry)
                current_entry = {'Client': line.split('Client:')[1].strip()}
            elif line.startswith('Conflict Type:'):
                current_entry['Conflict Type'] = line.split('Conflict Type:')[1].strip()
            elif line.startswith('Details:'):
                current_entry['Details'] = line.split('Details:')[1].strip()
        if current_entry:
            parsed_data.append(current_entry)

        return pd.DataFrame(parsed_data)
    except Exception as e:
        st.error(f"Error extracting conflict info: {e}")
        return pd.DataFrame(columns=['Client', 'Conflict Type', 'Details'])

# Streamlit app layout
st.title("Rolodex AI: Structured Conflict Check (Claude 3 Sonnet)")

# Data Overview Section
st.header("Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Matters Worked with", "10059")
with col2:
    st.metric("Data Updated from Clio API", "Last Update: 9/14/2024")

st.write("---")  # Adds a horizontal line for separation

st.write("Enter a client name to perform a conflict check:")

user_input = st.text_input("Client Name:", placeholder="e.g., 'Scale LLP'")

if user_input:
    progress_bar = st.progress(0)
    progress_bar.progress(10)
    matters_data = load_and_clean_data('combined_contact_and_matters.csv')
    if not matters_data.empty:
        progress_bar.progress(30)
        faiss_index, tfidf = create_vector_index(matters_data)
        if faiss_index is not None and tfidf is not None:
            progress_bar.progress(50)
            conflict_df = extract_conflict_info(matters_data, user_input, faiss_index, tfidf)
            progress_bar.progress(90)
            st.write("### Conflict Check Results:")
            if not conflict_df.empty:
                st.table(conflict_df)
            else:
                st.write("No potential conflicts or relevant information found.")
            progress_bar.progress(100)
        else:
            st.error("Failed to create vector index. Please check your FAISS installation.")
    else:
        st.error("Failed to load data.")
    progress_bar.empty()
