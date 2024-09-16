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

def extract_conflict_info(data, client_name, faiss_index, tfidf):
    # First, check for an exact match in the Client Name field
    exact_match = data[data['Client Name'].str.lower() == client_name.lower()]
    
    if not exact_match.empty:
        client_info = exact_match.iloc[0]
        conflict_message = f"Conflict found! Scale LLP has previously worked with the client: {client_name}"
        client_details = {
            'Client Name': client_info['Client Name'],
            'Matter': client_info['Matter'],
            'Phone Number': client_info.get('Primary Phone Number', 'N/A'),
            'Email Address': client_info.get('Primary Email Address', 'N/A')
        }
        st.write("Debug: Exact client match found")
    else:
        # If no exact match, proceed with the similarity search
        query_vec = tfidf.transform([client_name]).toarray().astype(np.float32)
        _, I = faiss_index.search(query_vec, k=50)  # Increased to 50 for more potential matches
        
        relevant_data = data.iloc[I[0]]

        st.write(f"Debug: Searching for similar clients to: {client_name}")
        st.write(f"Debug: Number of relevant data points: {len(relevant_data)}")

        if relevant_data.empty:
            st.write("Debug: No relevant data found")
            return None, None, None

        # Check if the firm has worked with a similar client before
        prior_work = relevant_data[
            relevant_data['Client Name'].str.contains(client_name, case=False, na=False) |
            relevant_data['Matter'].str.contains(client_name, case=False, na=False) |
            relevant_data['Matter Description'].str.contains(client_name, case=False, na=False)
        ]
        
        st.write(f"Debug: Number of prior work matches: {len(prior_work)}")

        if not prior_work.empty:
            client_info = prior_work.iloc[0]
            conflict_message = f"Potential conflict found! Scale LLP has previously worked with a similar client: {client_info['Client Name']}"
            client_details = {
                'Client Name': client_info['Client Name'],
                'Matter': client_info['Matter'],
                'Phone Number': client_info.get('Primary Phone Number', 'N/A'),
                'Email Address': client_info.get('Primary Email Address', 'N/A')
            }
            st.write("Debug: Similar client details found:", client_details)
        else:
            conflict_message = "No direct conflict found with the client."
            client_details = None
            st.write("Debug: No client details found")

    # Use the exact match or relevant data for further analysis
    analysis_data = exact_match if not exact_match.empty else relevant_data

    # Display the first few rows of analysis_data for debugging
    st.write("Debug: First few rows of analysis data:")
    st.write(analysis_data.head().to_string())

    # Analyze for potential opponents and business owners
    messages = [
        {"role": "system", "content": "You are a legal assistant tasked with identifying potential opponents, business owners, and analyzing matter descriptions related to a client."},
        {"role": "user", "content": f"""Analyze the following data for the client '{client_name}'. Identify:
1. Direct opponents of the client (look for 'v.', 'vs', or similar indicators in the Matter or Matter Description)
2. Potential opponents of the client based on the context of the matter
3. Any mentioned business owners related to the client

For each identified item, provide the following in a structured format:
- Type: [Direct Opponent], [Potential Opponent], or [Business Owner]
- Name: [Name of the opponent or business owner]
- Details: [Relevant details about the opponent or business owner, including the reasoning for potential opponents]

Pay special attention to matter descriptions containing 'v.' or 'vs' and provide a clear recommendation on who the potential opponent client might be in these cases.

Here's the relevant data:

{analysis_data.to_string()}

Provide your analysis in a structured format that can be easily converted to a table."""}
    ]

    claude_response = call_claude(messages)
    if not claude_response:
        st.write("Debug: No response from Claude")
        return conflict_message, client_details, None

    st.write("Debug: Claude's response:", claude_response)

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
    
    st.write("Debug: Parsed additional info:", additional_info.to_string())
    
    return conflict_message, client_details, additional_info

# Streamlit app layout
st.title("Rolodex AI: Detailed Conflict Check (Claude 3 Sonnet) - Enhanced Version")

# Data Overview Section
st.header("Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Matters Worked with", "10059")
with col2:
    st.metric("Data Updated from Clio API", "Last Update: 9/14/2024")

st.write("---")  # Adds a horizontal line for separation

st.write("Enter a client name to perform a conflict check:")

user_input = st.text_input("Client Name:", placeholder="e.g., 'Land On Top LLC'")

if user_input:
    progress_bar = st.progress(0)
    progress_bar.progress(10)
    matters_data = load_and_clean_data('combined_contact_and_matters.csv')
    if not matters_data.empty:
        progress_bar.progress(30)
        faiss_index, tfidf = create_vector_index(matters_data)
        progress_bar.progress(50)
        conflict_message, client_details, additional_info = extract_conflict_info(matters_data, user_input, faiss_index, tfidf)
        progress_bar.progress(90)
        
        st.write("### Conflict Check Results:")
        st.write(conflict_message)
        
        if client_details:
            st.write("#### Client Details:")
            for key, value in client_details.items():
                st.write(f"**{key}:** {value}")
        
        if additional_info is not None and not additional_info.empty:
            st.write("#### Potential Opponents, Direct Opponents, and Business Owners:")
            st.table(additional_info)
        else:
            st.write("No potential opponents, direct opponents, or business owners identified.")
        
        progress_bar.progress(100)
    else:
        st.error("Failed to load data.")
    progress_bar.empty()

    # Add a section to display raw data for debugging
    st.write("### Raw Data (for debugging):")
    st.write(matters_data[matters_data['Client Name'].str.contains(user_input, case=False, na=False)].head(10))
