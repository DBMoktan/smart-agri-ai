import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables (API Key)
load_dotenv()

# Configuration
DB_PATH = "data/vector_db/"

def get_rag_chain():
    """
    Initializes and returns a RetrievalQA chain.
    """
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Connect to the existing Vector Database
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector DB not found at {DB_PATH}. Run ingest.py first.")
        
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 3. Initialize Groq LLM (Llama 3.3)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2, # Low temperature for factual agricultural advice
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. Create a Custom Prompt (The "Instructions")
    prompt_template = """You are an expert Agricultural Assistant for Nepal. 
    Use the following pieces of retrieved context to answer the user's question. 
    If the context doesn't contain the answer, say you don't know, don't try to make up an answer.
    Answer in a professional and helpful tone.

    Context: {context}
    Question: {question}

    Expert Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # 5. Build the Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Pull top 3 relevant chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

if __name__ == "__main__":
    # Quick test
    try:
        chain = get_rag_chain()
        query = "What are the main constraints for maize production in Nepal?"
        print(f"\nUser Query: {query}")
        
        response = chain.invoke({"query": query})
        
        print("\n--- AI Response ---")
        print(response["result"])
        
        print("\n--- Sources Used ---")
        for doc in response["source_documents"]:
            print(f"Source: {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
            
    except Exception as e:
        print(f"Error: {e}")
