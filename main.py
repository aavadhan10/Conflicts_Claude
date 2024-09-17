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

def create_vector_index(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Matter Description'].fillna(''))
    
    faiss_index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
    faiss_index.add(np.array(tfidf_matrix.toarray(), dtype=np.float32))
    
    return faiss_index, tfidf

def extract_conflict_info(data, client_name=None, client_email=None, client_phone=None, faiss_index=None, tfidf=None):
    # Check for exact matches in any of the provided fields
    exact_match = data[
        (data['Client Name'].str.lower() == client_name.lower() if client_name else False) |
        (data['Primary Email Address'].str.lower() == client_email.lower() if client_email else False) |
        (data['Primary Phone Number'] == client_phone if client_phone else False)
    ]
    
    if not exact_match.empty:
        client_info = exact_match.iloc[0]
        conflict_message = f"Conflict found! Scale LLP has previously worked with this client."
        client_details = {
            'Client Name': client_info['Client Name'],
            'Matter': client_info['Matter'],
            'Phone Number': client_info.get('Primary Phone Number', 'N/A'),
            'Email Address': client_info.get('Primary Email Address', 'N/A')
        }
    else:
        # If no exact match, proceed with similarity search
        query = client_name or client_email or client_phone
        query_vec = tfidf.transform([query]).toarray().astype(np.float32)
        _, I = faiss_index.search(query_vec, k=50)
        
        relevant_data = data.iloc[I[0]]

        if relevant_data.empty:
            return None, None, None

        # Check if the firm has worked with a similar client before
        prior_work = relevant_data[
            relevant_data['Client Name'].str.contains(query, case=False, na=False) |
            relevant_data['Matter'].str.contains(query, case=False, na=False) |
            relevant_data['Matter Description'].str.contains(query, case=False, na=False)
        ]

        if not prior_work.empty:
            client_info = prior_work.iloc[0]
            conflict_message = f"Potential conflict found! Scale LLP has previously worked with a similar client: {client_info['Client Name']}"
            client_details = {
                'Client Name': client_info['Client Name'],
                'Matter': client_info['Matter'],
                'Phone Number': client_info.get('Primary Phone Number', 'N/A'),
                'Email Address': client_info.get('Primary Email Address', 'N/A')
            }
        else:
            conflict_message = "No direct conflict found with the client."
            client_details = None

    analysis_data = exact_match if not exact_match.empty else relevant_data

    # Analyze for potential opponents, business owners, and acquisition parties
    if client_details:
        messages = [
            {"role": "system", "content": "You are a legal assistant tasked with identifying potential opponents, business owners, acquisition parties, and analyzing matter descriptions related to a client."},
            {"role": "user", "content": f"""Analyze the following data for the client. Identify:
            1. Direct opponents of the client (look for 'v.', 'vs', or similar indicators in the Matter or Matter Description)
            2. Potential opponents of the client based on the context of the matter
            3. Any mentioned business owners related to the client
            4. Acquisition parties mentioned in relation to the client or their matters

            For each identified item, provide the following in a structured format:
            - Type: [Direct Opponent], [Potential Opponent], [Business Owner], or [Acquisition Party]
            - Name: [Name of the opponent, business owner, or acquisition party]
            - Details: [Relevant details about the opponent, business owner, or acquisition party, including the reasoning for potential opponents]

            Pay special attention to matter descriptions containing 'v.' or 'vs', and also look for terms related to acquisitions, mergers, or ownership transitions.

            Here's the relevant data:

            {analysis_data.to_string()}

            Provide your analysis in a structured format that can be easily converted to a table."""}
        ]

        claude_response = call_claude(messages)
        if not claude_response:
            return conflict_message, client_details, None

        # Parse Claude's response into a structured format
        lines = claude_response.split('\n')
        parsed_data = []
        current_entry = {}
        for line in lines:
            if line.startswith('Type:'):
                if current_entry:
                    parsed_data.append(current_entry)
                current_entry = {'Type': line.split('Type:')[1].strip()}
            elif line.startswith('Name:'):
                current_entry['Name'] = line.split('Name:')[1].strip()
            elif line.startswith('Details:'):
                current_entry['Details'] = line.split('Details:')[1].strip()
        if current_entry:
            parsed_data.append(current_entry)

        additional_info = pd.DataFrame(parsed_data)
    
        return conflict_message, client_details, additional_info
    else:
        return conflict_message, client_details, None

# Load data and create index (this should be done once and cached)
@st.cache_resource
def load_data_and_create_index():
    matters_data = load_and_clean_data('combined_contact_and_matters.csv')
    faiss_index, tfidf = create_vector_index(matters_data)
    return matters_data, faiss_index, tfidf

matters_data, faiss_index, tfidf = load_data_and_create_index()

# Sidebar for Data Overview
st.sidebar.header("📊 Data Overview")
st.sidebar.metric("Number of Matters Worked with", "10,061")
st.sidebar.metric("Data Updated from Clio API", "Last Update: 9/17/2024")

# Main content
st.title("Scale LLP Conflict Check System")

# Input fields
client_name = st.text_input("Enter Client's Full Name")
client_email = st.text_input("Enter Client's Email")
client_phone = st.text_input("Enter Client's Phone Number")

if st.button("Check for Conflict"):
    if client_name or client_email or client_phone:
        with st.spinner("Performing conflict check..."):
            conflict_message, client_details, additional_info = extract_conflict_info(
                matters_data, 
                client_name=client_name, 
                client_email=client_email, 
                client_phone=client_phone, 
                faiss_index=faiss_index, 
                tfidf=tfidf
            )
            
            st.write("### Conflict Check Results:")
            st.write(conflict_message)
            
            if client_details:
                st.write("#### Client Details:")
                for key, value in client_details.items():
                    st.write(f"**{key}:** {value}")
            
                if additional_info is not None and not additional_info.empty:
                    st.write("#### Potential Opponents, Direct Opponents, Business Owners, and Acquisition Parties:")
                    st.table(additional_info.reset_index(drop=True))  # Remove the index from the table
            else:
                st.write("No additional parties or conflicts identified.")
    else:
        st.error("Please enter at least one field (Name, Email, or Phone Number)")
