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
        st.write("Debug: Exact client match found")
    else:
        # If no exact match, proceed with similarity search
        query = client_name or client_email or client_phone
        query_vec = tfidf.transform([query]).toarray().astype(np.float32)
        _, I = faiss_index.search(query_vec, k=50)
        
        relevant_data = data.iloc[I[0]]

        st.write(f"Debug: Searching for similar clients")
        st.write(f"Debug: Number of relevant data points: {len(relevant_data)}")

        if relevant_data.empty:
            st.write("Debug: No relevant data found")
            return None, None, None

        # Check if the firm has worked with a similar client before
        prior_work = relevant_data[
            relevant_data['Client Name'].str.contains(query, case=False, na=False) |
            relevant_data['Matter'].str.contains(query, case=False, na=False) |
            relevant_data['Matter Description'].str.contains(query, case=False, na=False)
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
        {"role": "user", "content": f"""Analyze the following data for the client. Identify:
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

# Load data and create index (this should be done once and cached)
@st.cache_resource
def load_data_and_create_index():
    matters_data = load_and_clean_data('combined_contact_and_matters.csv')
    faiss_index, tfidf = create_vector_index(matters_data)
    return matters_data, faiss_index, tfidf

matters_data, faiss_index, tfidf = load_data_and_create_index()

# Sidebar
st.sidebar.title("📊 Data Overview")

# Display number of matters worked with
num_matters = len(data)
st.sidebar.markdown(f"<h2 style='color: #4CAF50;'>Number of Matters Worked with: {num_matters}</h2>", unsafe_allow_html=True)

# Add a banner or button for data update info
st.sidebar.markdown(
    "<div style='background-color: #f0f0f5; padding: 10px; border-radius: 5px; border: 1px solid #ccc;'>"
    "<strong>Data Updated from Clio API</strong><br>Last Update: <strong>9/14/2024</strong>"
    "</div>", unsafe_allow_html=True
)

# Main content
st.title("Scale LLP Conflict Check System with Relationship Graph")

# Input fields
client_name = st.text_input("Enter Client's Full Name")
client_email = st.text_input("Enter Client's Email")
client_phone = st.text_input("Enter Client's Phone Number")

col1, col2 = st.columns(2)
with col1:
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
                    st.write("#### Potential Opponents, Direct Opponents, and Business Owners:")
                    st.table(additional_info)
                else:
                    st.write("No potential opponents, direct opponents, or business owners identified.")
        else:
            st.error("Please enter at least one field (Name, Email, or Phone Number)")

with col2:
    if st.button("Create Relationship Graph"):
        if client_name or client_email or client_phone:
            st.write("Creating relationship graph...")
            # Add your relationship graph creation logic here
            st.write("Relationship graph functionality not implemented yet.")
        else:
            st.error("Please enter at least one field (Name, Email, or Phone Number)")

# Placeholder for relationship graph
st.header("Relationship Graph")
st.write("Relationship graph will be displayed here once implemented.")
