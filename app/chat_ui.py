import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agri-Smart AI | Nepal",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.8rem; margin-bottom: 1rem; display: flex;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .chat-message.user {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
    }
    .chat-message.bot {
        background-color: #e8f5e9;
        border-left: 5px solid #1b5e20;
    }
    .chat-icon {
        width: 40px; height: 40px; border-radius: 50%; object-fit: cover; margin-right: 1rem;
    }
    .source-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
    h1 {
        color: #1b5e20;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910791.png", width=100)
    st.title("Agri-Smart AI")
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This AI Assistant is powered by RAG (Retrieval-Augmented Generation) 
    and specialized in Nepalese Agriculture. 
    
    It uses documents from the Ministry of Agriculture and Livestock Development, Nepal.
    """)
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("🌱 Agri-Smart AI: Nepal's Knowledge Hub")
st.caption("Ask me anything about maize production, soil fertility, or agricultural statistics in Nepal.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:** {source['metadata'].get('source', 'Unknown')}")
                    st.markdown(f"*{source['content'][:200]}...*")

# Chat input
if prompt := st.chat_input("How can I improve maize yield in hilly regions?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🚜 *Consulting the knowledge base...*")
        
        try:
            response = requests.post(API_URL, json={"query": prompt})
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                message_placeholder.markdown(answer)
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                if sources:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"**Source {idx+1}:** {source['metadata'].get('source', 'Unknown')}")
                            st.markdown(f"*{source['content'][:200]}...*")
            else:
                st.error(f"API Error: {response.status_code}")
                message_placeholder.markdown("Sorry, I encountered an error while processing your request.")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            message_placeholder.markdown("Failed to connect to the backend server. Is it running?")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ for Nepalese Farmers | Powered by Groq & LangChain</p>", unsafe_allow_html=True)
