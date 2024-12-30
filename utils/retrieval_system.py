import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Create directories if they do not exist
os.makedirs("data/processed_data", exist_ok=True)

# Load processed embeddings
train_embeddings = np.load('D:/RAG-GenAI/data/processed_data/train_embeddings.npy')
valid_embeddings = np.load('D:/RAG-GenAI/data/processed_data/valid_embeddings.npy')

# Load the dataset (assumes preprocessed)
train_data = pd.read_csv('D:/RAG-GenAI/data/raw_data/train.csv')  # Adjust paths if needed

# Initialize FAISS index
embedding_dim = train_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(train_embeddings)  # Add train embeddings to FAISS index

print(f"FAISS index contains {index.ntotal} documents.")

# Define function to retrieve similar documents
def retrieve_similar_documents(query_embedding, k=5):
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)  # Search the FAISS index
    return distances, indices

# Helper to fetch documents
def get_documents_by_indices(dataframe, indices):
    return dataframe.iloc[indices.flatten()]

# Example usage
if __name__ == "__main__":
    # Load SentenceTransformer model for embedding queries
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Create embedding for a sample query
    query = "What is AI?"
    query_embedding = model.encode([query])[0]  # Generate query embedding

    # Retrieve top 5 similar documents
    distances, indices = retrieve_similar_documents(query_embedding, k=5)
    print("Top 5 similar document distances:", distances)
    print("Top 5 similar document indices:", indices)

    # Fetch and print the actual documents
    retrieved_docs = get_documents_by_indices(train_data, indices)
    print("Retrieved Documents:")
    print(retrieved_docs[['query', 'finalpassage']])  # Adjust columns as needed
    faiss.write_index(index, "data/processed_data/faiss_index.idx")
# To load it later:
    index = faiss.read_index("data/processed_data/faiss_index.idx")
