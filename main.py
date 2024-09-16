import streamlit as st
import pandas as pd
import numpy as np
#import networkx as nx
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
    
    # Check for direct match in the 'Client Name' field
    exact_match = data[
        (data['Client Name'].str.lower() == client_name if client_name else False)
    ]
    
    # If an exact match is found, it's considered a conflict
    if not exact_match.empty:
        client_info = exact_match.iloc[0]
        conflict_message = f"Conflict found! Scale LLP has previously worked with this client."
        client_details = {
            'Client Name': client_info['Client Name'],
            'Matter': client_info['Matter'],
            'Phone Number': client_info.get('Primary Phone Number', 'N/A'),
            'Email Address': client_info.get('Primary Email Address', 'N/A')
        }
        return conflict_message, client_details, None

    # No exact match, so we proceed with the similarity search
    query = client_name or client_email or client_phone
    if query:
        query_vec = tfidf.transform([query]).toarray().astype(np.float32)
        _, I = faiss_index.search(query_vec, k=50)
        
        relevant_data = data.iloc[I[0]]
        
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
            return conflict_message, client_details, None
    
    # No conflicts found
    conflict_message = "No direct conflict found with the client."
    return conflict_message, None, None

# Create a relationship graph
def create_relationship_graph(data, client_name):
    G = nx.Graph()

    G.add_node(client_name, label=client_name)

    for index, row in data.iterrows():
        client = row.get('Client Name', '')
        if client != client_name and client:
            G.add_node(client, label=client)
            G.add_edge(client_name, client, label="Worked together on a case")

    return G

# Draw and display relationship graph
def draw_relationship_graph(graph):
    net = Network(height="600px", width="100%", notebook=True)
    net.from_nx(graph)
    net.show("relationship_graph.html")
    
    HtmlFile = open("relationship_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=600)

# Load data and create index
@st.cache_resource
def load_data_and_create_index():
    matters_data = load_and_clean_data('combined_contact_and_matters.csv')
    faiss_index, tfidf = create_vector_index(matters_data)
    return matters_data, faiss_index, tfidf

matters_data, faiss_index, tfidf = load_data_and_create_index()

# Sidebar for Data Overview
st.sidebar.header("📊 Data Overview")
st.sidebar.metric("Number of Matters Worked with", "10,059")
st.sidebar.metric("Data Updated from Clio API", "Last Update: 9/14/2024")

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
                
                # Display the conflict message in green if a conflict is found
                if "Conflict found" in conflict_message or "Potential conflict found" in conflict_message:
                    st.success(conflict_message)
                else:
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
            graph = create_relationship_graph(matters_data, client_name)
            draw_relationship_graph(graph)
        else:
            st.error("Please enter at least one field (Name, Email, or Phone Number)")

# Placeholder for relationship graph
st.header("Relationship Graph")
st.write("Relationship graph will be displayed here once implemented.")
