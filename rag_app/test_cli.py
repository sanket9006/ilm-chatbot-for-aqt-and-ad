import sys
import os

# Add the current directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline

def main():
    print("--- E-commerce RAG CLI Test ---")
    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    
    print("Initializing pipeline (this may take a few seconds to load the model)...")
    pipeline = RAGPipeline(data_path)
    print("Pipeline Ready!\n")

    test_queries = [
        "Which air purifier is best for allergies?",
        "Difference between AquaTru Classic and Connect?",
        "How often should I change filters for AirDoctor 3000?",
        "Tell me about the Under Sink AquaTru model."
    ]

    for query in test_queries:
        print(f"User: {query}")
        result = pipeline.ask(query)
        print(f"Assistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("-" * 30)

if __name__ == "__main__":
    main()
