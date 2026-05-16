import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def get_rag_chain():
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load Vector Store
    vector_db = Chroma(
        persist_directory="data/vector_db",
        embedding_function=embeddings
    )
    
    # 3. Initialize LLM (Groq) - Using 8B model for higher rate limits
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.1-8b-instant"
    )
    
    # 4. Define Sharp Professional Prompt
    prompt_template = """
    You are an expert Agricultural Assistant for Nepal. Use the following context to answer the user's question precisely.
    
    RULES:
    1. Be concise and direct. Answer the question in the first sentence if possible.
    2. Use bullet points for statistics or lists.
    3. If the answer is not in the context, say: "I'm sorry, but my current records do not contain specific information on this. Please consult a local agricultural expert."
    4. Do not mention "based on the provided context" - just give the answer.
    
    CONTEXT: {context}
    
    QUESTION: {question}
    
    ANSWER:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 5. Create Chain with k=5 for better context coverage
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

if __name__ == "__main__":
    # Test the chain
    chain = get_rag_chain()
    query = "What is the average yield of maize in Nepal?"
    response = chain.invoke(query)
    print(f"Query: {query}")
    print(f"Answer: {response['result']}")
