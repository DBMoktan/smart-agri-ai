# 🌾 Agri-Smart AI: RAG-Powered Knowledge Assistant for Nepal

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-orange.svg)](https://www.langchain.com/)
[![Ragas](https://img.shields.io/badge/Ragas-Evaluation-red.svg)](https://ragas.io/)

**Agri-Smart AI** is a professional Retrieval-Augmented Generation (RAG) system designed to bridge the knowledge gap for Nepalese farmers and agricultural stakeholders. By indexing over **1,150 pages** of national statistics, soil management research, and crop cultivation guides, the system provides high-precision, evidence-based agricultural advice.

---

## 📈 Performance Benchmarks (via Ragas)
We evaluated the system's performance across critical agricultural queries. The results demonstrate high reliability and factual precision.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Faithfulness** | **97.5%** | Near-zero hallucination rate; responses are grounded in verified documents. |
| **Answer Relevancy** | **89.9%** | Highly direct and helpful responses tailored to specific agricultural queries. |
| **Data Coverage** | **1,153 Pages** | 2,901 specialized text chunks indexed from Nepalese agricultural research. |

---

## 🏗️ System Architecture
The system utilizes a state-of-the-art modular RAG architecture:

1.  **Ingestion**: PDFs are processed using `PyPDF`, split into semantically meaningful chunks, and embedded using `HuggingFace (all-MiniLM-L6-v2)`.
2.  **Storage**: Chunks are stored in a persistent **ChromaDB** vector store.
3.  **Retrieval**: Multi-document retrieval with **k=5** to ensure comprehensive context coverage.
4.  **Inference**: Responses generated via **Groq LPU (Llama-3.1-8B)** for ultra-low latency and high intelligence.
5.  **Evaluation**: Automated quality assurance via the **Ragas** framework.

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   Groq API Key (Available at [console.groq.com](https://console.groq.com/))

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/yourusername/agri-smart-ai.git
cd agri-smart-ai

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Ingestion
Place your agricultural PDFs in `data/raw/` and run:
```powershell
python src/ingest.py
```

### 4. Running the Application
**Terminal 1 (Backend API):**
```powershell
python -m uvicorn app.main:app --reload
```

**Terminal 2 (Frontend UI):**
```powershell
streamlit run app/chat_ui.py
```

### 5. Running Evaluation
To verify performance on your machine:
```powershell
python evaluate.py
```

---

## 🛠️ Tech Stack
*   **LLM Engine**: Groq (Llama 3.1)
*   **Orchestration**: LangChain
*   **Vector Database**: ChromaDB
*   **Backend**: FastAPI
*   **UI**: Streamlit
*   **Evaluation**: Ragas

---

*Built with ❤️ for the Nepalese Agricultural Community.*
