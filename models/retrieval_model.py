import os
import faiss
import numpy as np
import json

class RetrievalModel:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None

    def load_index(self):
        """Load the FAISS index from the specified path."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"FAISS index loaded successfully with dimension: {self.index.d}")

    def load_metadata(self):
        """Load metadata corresponding to the FAISS index."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print("Metadata loaded successfully.")

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5):
        """Retrieve top-k results from the FAISS index for a given query vector."""
        if self.index is None or self.metadata is None:
            raise ValueError("Index and metadata must be loaded before retrieval.")

        # Debugging: Print query vector dimensions
        print(f"Query vector shape: {query_vector.shape}")
        assert query_vector.shape[1] == self.index.d, (
            f"Dimension mismatch: query_vector ({query_vector.shape[1]}) vs FAISS index ({self.index.d})"
        )

        distances, indices = self.index.search(query_vector, top_k)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Handle empty results
                continue
            metadata_entry = self.metadata.get(str(idx), "No metadata found")
            results.append({
                "distance": dist,
                "metadata": metadata_entry
            })

        return results

if __name__ == "__main__":
    INDEX_PATH = "../data/processed_data/faiss_index.idx"
    METADATA_PATH = "../data/processed_data/metadata.json"

    # Initialize the retrieval model
    retrieval_model = RetrievalModel(INDEX_PATH, METADATA_PATH)

    # Load the index and metadata
    retrieval_model.load_index()
    retrieval_model.load_metadata()

    # Example query vector with corrected dimensions
    example_query_vector = np.random.rand(1, 384).astype('float32')

    # Retrieve top-k results
    try:
        top_k_results = retrieval_model.retrieve(example_query_vector, top_k=5)

        # Display the results
        for result in top_k_results:
            print(f"Distance: {result['distance']}, Metadata: {result['metadata']}")
    except AssertionError as e:
        print(f"Error: {e}")

