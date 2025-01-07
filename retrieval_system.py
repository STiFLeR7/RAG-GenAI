import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_file, metadata_file):
    
    print("Loading FAISS index...")
    index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as file:
        metadata = pickle.load(file)
    print("FAISS index and metadata loaded successfully.")
    return index, metadata

def query_index(index, metadata, query, model, top_k=5):
    
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [{"distance": distances[0][i], "chunk": metadata["chunks"][indices[0][i]]} for i in range(top_k)]
    return results

if __name__ == "__main__":
    # File paths
    index_file = "D:/RAG-GenAI/Physics-FAISS.idx"
    metadata_file = "D:/RAG-GenAI/Physics-FAISS_metadata.pkl"
    model_name = "all-MiniLM-L6-v2"

    # Load FAISS index and metadata
    index, metadata = load_faiss_index(index_file, metadata_file)

    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Example queries
    queries = [
        "What is Newton's second law of motion?",
        "Explain the photoelectric effect.",
        "What is the theory of relativity?"
    ]

    # Perform retrieval
    for query in queries:
        print(f"\nQuery: {query}")
        results = query_index(index, metadata, query, model, top_k=5)
        for result in results:
            print(f"Distance: {result['distance']:.2f}, Chunk: {result['chunk']}")
