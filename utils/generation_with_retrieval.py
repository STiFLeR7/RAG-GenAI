from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load preprocessed data and FAISS index
train_embeddings = np.load('D:/RAG-GenAI/data/processed_data/train_embeddings.npy')  # Adjust path if needed
train_data = pd.read_csv('D:/RAG-GenAI/data/raw_data/train.csv')  # Adjust path if needed

# Initialize FAISS index
embedding_dim = train_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(train_embeddings)  # Add train embeddings to FAISS index

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load SentenceTransformer model for embedding queries
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define function to retrieve similar documents
def retrieve_similar_documents(query_embedding, k=5):
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)  # Search the FAISS index
    return distances, indices

# Helper to fetch documents by indices
def get_documents_by_indices(dataframe, indices):
    return dataframe.iloc[indices.flatten()]

# Example query
query = "What is AI?"
query_embedding = sentence_model.encode([query])[0]  # Generate query embedding

# Retrieve top 5 similar documents
distances, indices = retrieve_similar_documents(query_embedding, k=5)
print("Top 5 similar document distances:", distances)
print("Top 5 similar document indices:", indices)

# Fetch and print the actual documents
retrieved_docs = get_documents_by_indices(train_data, indices)
print("Retrieved Documents:")
print(retrieved_docs[['query', 'finalpassage']])  # Adjust columns as needed

# Concatenate retrieved documents' text to form a context for the generative model
retrieved_context = " ".join(retrieved_docs['finalpassage'].values)

# Prepare input text for T5 (e.g., as a question-answer format)
input_text = f"Answer this question: {query} Context: {retrieved_context}"

# Tokenize and generate output
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)
