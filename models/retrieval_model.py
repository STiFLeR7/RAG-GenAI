import os
import faiss
import numpy as np
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

class RetrievalModel:
    def __init__(self, index_path: str, metadata_path: str, generation_model_name: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.tokenizer = T5Tokenizer.from_pretrained(generation_model_name)
        self.generation_model = T5ForConditionalGeneration.from_pretrained(generation_model_name)

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

    def generate_response(self, query: str, top_k: int = 5):
        """Generate a response based on retrieved documents and the query."""
        query_vector = self.encode_query(query)
        retrieved_docs = self.retrieve(query_vector, top_k)

        # Concatenate the top-k documents into a context
        context = " ".join([doc['metadata']['finalpassage'] for doc in retrieved_docs if 'metadata' in doc])
        input_text = f"Query: {query} Context: {context}"

        # Tokenize and generate response
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = self.generation_model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response, retrieved_docs

    def encode_query(self, query: str):
        """Encode a query into a vector using the retrieval model."""
        # Example placeholder for encoding logic, replace with your actual encoder
        return np.random.rand(1, self.index.d).astype('float32')

if __name__ == "__main__":
    INDEX_PATH = "../data/processed_data/faiss_index.idx"
    METADATA_PATH = "../data/processed_data/metadata.json"
    GENERATION_MODEL_NAME = "t5-small"

    # Initialize the retrieval model
    retrieval_model = RetrievalModel(INDEX_PATH, METADATA_PATH, GENERATION_MODEL_NAME)

    # Load the index, metadata, and generation model
    retrieval_model.load_index()
    retrieval_model.load_metadata()

    # Example query
    user_query = "What is AI?"

    # Generate response
    response, retrieved_docs = retrieval_model.generate_response(user_query, top_k=5)

    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print(f"Distance: {doc['distance']}, Metadata: {doc['metadata']}")

    print("\nGenerated Response:", response)
