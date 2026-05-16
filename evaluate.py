import os
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from src.rag_engine import get_rag_chain

# Load environment variables
load_dotenv()

def run_evaluation():
    print("🚀 Initializing Agri-Smart Evaluation Pipeline...")
    
    # 1. Initialize the Judge LLM (using a faster model to avoid rate limits)
    evaluator_llm = ChatGroq(model_name="llama-3.1-8b-instant")
    
    # 2. Initialize the Judge Embeddings (using our project's HuggingFace model)
    # This avoids the need for an OpenAI API key
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Initialize our RAG Chain
    rag_chain = get_rag_chain()
    
    # 4. Define Test Dataset
    test_questions = [
        "What is the average yield of maize in Nepal according to the latest stats?",
        "What are the main soil fertility management challenges in Nepal?",
        "How much area was covered by maize cultivation in 2022/23?",
        "What are the recommendations for fertilizer application in maize farming?",
        "Which province in Nepal has the highest maize production?"
    ]
    
    print(f"📋 Running RAG on {len(test_questions)} test questions...")
    results = []
    for query in test_questions:
        print(f"  -> Testing: {query}")
        response = rag_chain.invoke(query)
        results.append({
            "question": query,
            "answer": response["result"],
            "contexts": [doc.page_content for doc in response["source_documents"]],
        })
    
    dataset = Dataset.from_list(results)
    
    # 5. Perform Evaluation
    print("📊 Calculating Ragas Metrics (Using Groq & HuggingFace Judges)...")
    # We explicitly pass our LLM and Embeddings to the metrics
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    # 6. Save and Display Results
    df_results = score.to_pandas()
    output_path = "data/evaluation_report.csv"
    os.makedirs("data", exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE")
    print(f"📈 Overall Summary:\n{score}")
    print(f"📄 Detailed report saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"❌ Evaluation Failed: {e}")
