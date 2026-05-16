import os
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_groq import ChatGroq

from datasets import Dataset
from src.rag_engine import get_rag_chain

# Load environment variables
load_dotenv()

def run_evaluation():
    print("🚀 Initializing Agri-Smart Evaluation Pipeline...")
    
    # Initialize the judge LLM (using Groq)
    evaluator_llm = ChatGroq(model_name="llama3-70b-8192")
    rag_chain = get_rag_chain()
    
    # 2. Define a Test Dataset (Questions based on your PDFs)
    # In a production scenario, these would be dozens of expert-verified pairs
    test_questions = [
        "What is the average yield of maize in Nepal according to the latest stats?",
        "What are the main soil fertility management challenges in Nepal?",
        "How much area was covered by maize cultivation in 2022/23?",
        "What are the recommendations for fertilizer application in maize farming?",
        "Which province in Nepal has the highest maize production?"
    ]
    
    # Optional: Ground truths if you want to measure Recall exactly
    # For now, we will focus on Faithfulness and Relevancy (which don't strictly require ground truths)
    
    print(f"📋 Running RAG on {len(test_questions)} test questions...")
    
    results = []
    for query in test_questions:
        print(f"  -> Testing: {query}")
        # Run the chain
        response = rag_chain.invoke(query)
        
        # Extract data for Ragas
        results.append({
            "question": query,
            "answer": response["result"],
            "contexts": [doc.page_content for doc in response["source_documents"]],
            # "ground_truth": "..." # Optional
        })
    
    # 3. Convert to HuggingFace Dataset format for Ragas
    dataset = Dataset.from_list(results)
    
    # 4. Perform Evaluation
    print("📊 Calculating Ragas Metrics (Faithfulness, Relevancy, Context)...")
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm
    )
    
    # 5. Save and Display Results
    df_results = score.to_pandas()
    output_path = "data/evaluation_report.csv"
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
