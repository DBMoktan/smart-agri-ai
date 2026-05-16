# Agri-Smart AI: A RAG-Powered Knowledge Assistant 🌾🤖

Agri-Smart AI is a specialized Retrieval-Augmented Generation (RAG) assistant designed to provide evidence-based agricultural advice to farmers and researchers in Nepal. Unlike general-purpose AI, this system searches through curated agricultural research papers, soil guidelines, and government reports to deliver accurate, localized insights.

## 🚀 Vision
To bridge the gap between complex agricultural research and practical on-field application using state-of-the-art LLMs and high-quality local knowledge bases.

## 🛠️ Tech Stack
- **LLM Engine:** Groq / Ollama (Llama 3)
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Backend:** FastAPI (Async & Streaming)
- **Frontend:** Streamlit (Modern UI)
- **Deployment:** Docker & Hugging Face Spaces

## 📂 Project Structure
```text
agri-smart-ai/
├── app/
│   ├── main.py          # FastAPI backend
│   └── chat_ui.py       # Streamlit frontend
├── src/
│   ├── ingest.py        # PDF processing & Embedding logic
│   └── rag_engine.py    # LangChain retrieval logic
├── data/
│   ├── raw/             # Expert PDFs & Reports
│   └── vector_db/       # ChromaDB persistent storage
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

## 📅 Roadmap (7-Day Sprint)
- **Day 1:** Data Research & Project Setup
- **Day 2:** Vector DB & Embeddings
- **Day 3:** RAG Pipeline Logic
- **Day 4:** FastAPI Backend with Streaming
- **Day 5:** Streamlit Modern Interface
- **Day 6:** Containerization & Deployment
- **Day 7:** Final Documentation & Showcase

## 🏃 How to Run
### 1. Set up Environment
Create a `.env` file and add your Groq API Key:
```bash
GROQ_API_KEY=your_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Data (Optional)
If you haven't already populated the vector database:
```bash
python src/ingest.py
```

### 4. Start the Backend API
```bash
uvicorn app.main:app --reload
```

### 5. Start the Streamlit UI
In a new terminal:
```bash
streamlit run app/chat_ui.py
```

---
Developed with ❤️ for the Nepalese Agricultural Community.
