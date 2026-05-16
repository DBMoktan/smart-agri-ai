import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Agri-Smart AI | Knowledge Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8fbf8;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e3d2f;
        color: white;
    }
    
    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }

    /* Bubble Citations */
    .source-tag {
        font-size: 0.8rem;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2329/2329113.png", width=100)
    st.title("Agri-Smart AI")
    st.markdown("---")
    st.markdown("### 🌾 About this Assistant")
    st.write("This AI is trained on specialized agricultural documents from Nepal, including soil management, maize cultivation, and national production statistics.")
    
    st.markdown("### 🛠️ Settings")
    use_citations = st.checkbox("Show Source Citations", value=True)
    clear_chat = st.button("Clear Conversation")
    
    st.markdown("---")
    st.info("Powered by **Groq LPU** & **LangChain** for ultra-fast agricultural insights.")

# --- Session State Initialization ---
if "messages" not in st.session_state or clear_chat:
    st.session_state.messages = []

# --- Main UI Header ---
st.title("🌾 Agri-Smart Knowledge Assistant")
st.caption("Expert agricultural advice based on Nepalese research and statistics.")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and use_citations:
            with st.expander("🔍 View Verified Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**[{idx+1}] {source['metadata'].get('source', 'Document')}**")
                    st.markdown(f"*{source['content'][:300]}...*")

# --- Chat Input & Logic ---
if prompt := st.chat_input("Ask me about farming, crops, or statistics in Nepal..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call FastAPI Backend
            response = requests.post(
                "http://localhost:8000/ask",
                json={"query": prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])
                
                # Typing effect simulation
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Handle Sources
                if sources and use_citations:
                    with st.expander("🔍 View Verified Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"**[{idx+1}] {source['metadata'].get('source', 'Document')}**")
                            st.markdown(f"*{source['content'][:300]}...*")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
            else:
                st.error("The backend returned an error. Please check if the API is working correctly.")
                
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            st.info("💡 Make sure your FastAPI server is running with: `uvicorn app.main:app --reload`")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Expert Agricultural Intelligence for Nepal</p>", unsafe_allow_html=True)
