import faiss
import pickle
import numpy as np
import os

def create_faiss_index(embeddings_file, index_file):
    
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        return

    try:
        # Load embeddings
        with open(embeddings_file, 'rb') as file:
            data = pickle.load(file)
        embeddings = np.array(data["embeddings"])
        chunks = data["chunks"]

        # Create FAISS index
        dimension = embeddings.shape[1]
        print(f"Creating FAISS index with dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        index.add(embeddings)

        # Save index and metadata
        faiss.write_index(index, index_file)
        metadata_file = index_file.replace(".idx", "_metadata.pkl")
        with open(metadata_file, 'wb') as file:
            pickle.dump({"chunks": chunks}, file)

        print(f"FAISS index saved to {index_file}")
        print(f"Metadata saved to {metadata_file}")

    except Exception as e:
        print(f"An error occurred while creating the FAISS index: {e}")

if __name__ == "__main__":
    # Define paths
    embeddings_file = "D:/RAG-GenAI/Physics-embeddings.pkl"
    index_file = "D:/RAG-GenAI/Physics-FAISS.idx"

    # Create FAISS index
    create_faiss_index(embeddings_file, index_file)
