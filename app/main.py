import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import get_rag_chain

app = FastAPI(
    title="Agri-Smart AI API",
    description="Backend API for the Agricultural RAG Assistant",
    version="1.0.0"
)

# Initialize the RAG chain globally for efficiency
try:
    rag_chain = get_rag_chain()
except Exception as e:
    print(f"Error initializing RAG chain: {e}")
    rag_chain = None

class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

@app.get("/")
async def root():
    return {"message": "Welcome to Agri-Smart AI API. Use /ask for queries."}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized. Check server logs.")
    
    try:
        # Using invoke instead of __call__ as per modern LangChain practice
        response = rag_chain.invoke({"query": request.query})
        
        answer = response["result"]
        sources = [
            SourceDocument(content=doc.page_content, metadata=doc.metadata)
            for doc in response["source_documents"]
        ]
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
