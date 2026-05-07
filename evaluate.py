import os
import sys
import time

# Add project root to sys.path
PROJECT_DIR = r"c:\Users\amitv\OneDrive\Documents\Gravity\GEN AI\history-fact-bot"
sys.path.append(PROJECT_DIR)

from rag_chain import create_rag_chain, ask

def evaluate_model():
    """Evaluate the RAG system using golden questions."""
    print("--- MODEL EVALUATION PIPELINE ---")
    
    # Selection of suitable evaluation dataset (Golden Questions)
    test_cases = [
        {
            "question": "Who was Joan of Arc?",
            "expected_source": "hundred_years_war.txt",
            "keywords": ["French", "Orléans", "Saint"]
        },
        {
            "question": "What is the Magna Carta?",
            "expected_source": "magna_carta.txt",
            "keywords": ["1215", "Runnymede", "King John"]
        },
        {
            "question": "What was the main cause of the Cold War?",
            "expected_source": "modern_history.txt",
            "keywords": ["Soviet", "USA", "Nuclear"]
        }
    ]
    
    chain = create_rag_chain()
    
    results = []
    total_retrieval_score = 0
    
    print("\nRunning Evaluation...")
    for test in test_cases:
        print(f"\nEvaluating: '{test['question']}'")
        start_time = time.time()
        response = ask(chain, test["question"])
        latency = round(time.time() - start_time, 2)
        
        # Metric 1: Retrieval Success (Exact source match)
        sources = [os.path.basename(doc.metadata.get("source", "")) for doc in response.get("source_documents", [])]
        retrieval_success = test["expected_source"] in sources
        if retrieval_success:
            total_retrieval_score += 1
            
        # Metric 2: Fact Accuracy (Keyword presence)
        accuracy_score = sum(1 for kw in test["keywords"] if kw.lower() in response["answer"].lower()) / len(test["keywords"])
        
        print(f"- Latency: {latency}s")
        print(f"- Retrieval Correct: {'YES' if retrieval_success else 'NO'}")
        print(f"  (Sources found: {sources})")
        print(f"- Fact Accuracy Score: {round(accuracy_score * 100, 2)}%")
        
        results.append({
            "retrieval": retrieval_success,
            "accuracy": accuracy_score,
            "latency": latency
        })

    # Summary Metrics
    avg_latency = sum(r["latency"] for r in results) / len(results)
    retrieval_precision = (total_retrieval_score / len(test_cases)) * 100
    avg_accuracy = (sum(r["accuracy"] for r in results) / len(results)) * 100
    
    print("\n--- EVALUATION METRICS SUMMARY ---")
    print(f"1. Retrieval Precision (Top-K): {retrieval_precision}%")
    print(f"2. Mean Fact Accuracy (Grounded): {avg_accuracy}%")
    print(f"3. Mean Inference Latency: {avg_latency}s")
    print("---------------------------------")

if __name__ == "__main__":
    evaluate_model()
