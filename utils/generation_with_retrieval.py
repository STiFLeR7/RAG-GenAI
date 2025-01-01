import json
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import faiss
import pandas as pd

# Model and FAISS setup
retrieval_model = SentenceTransformer("models/retrieval_model")
generation_model_name = "models/generation_model"
generation_model = T5ForConditionalGeneration.from_pretrained(generation_model_name)
tokenizer = T5Tokenizer.from_pretrained(generation_model_name)

faiss_index_path = "data/processed_data/faiss_index.idx"
data_path = "data/processed_data/processed_data.json"

# Load FAISS index
faiss_index = faiss.read_index(faiss_index_path)
with open(data_path, "r") as f:
    data = json.load(f)

# Retrieval function
def retrieve_top_k(query, k=5):
    query_embedding = retrieval_model.encode(query).reshape(1, -1).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k)
    top_k_docs = [data[i]["finalpassage"] for i in indices[0]]
    return top_k_docs, distances, indices

# Generate response
def generate_response(query):
    top_docs, distances, indices = retrieve_top_k(query)
    context = " ".join(top_docs)
    input_text = f"Query: {query} Context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = generation_model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response, distances, indices

# Main execution
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response, distances, indices = generate_response(user_query)
    print("\nTop 5 similar document distances:", distances)
    print("Top 5 similar document indices:", indices)
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retrieve_top_k(user_query)[0], start=1):
        print(f"{i}: {doc}")
    print("\nGenerated Response:", response)
