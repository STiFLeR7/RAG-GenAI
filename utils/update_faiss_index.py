import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Paths and constants
INDEX_PATH = "D:/RAG-GenAIdata/processed_data/faiss_index.idx"
NEW_DATA_PATH = "D:/RAG-GenAI/data/raw_data/new_data.csv"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64

# Load the Sentence Transformer model
print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

# Function to load the FAISS index
def load_faiss_index():
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_PATH)
    else:
        print("Creating a new FAISS index...")
        index = faiss.IndexFlatL2(768)  # Ensure dimensions match the embeddings
    return index

# Function to preprocess and embed new data
def process_new_data(file_path):
    print("Processing new data...")
    if not os.path.exists(file_path):
        print(f"No new data found at {file_path}. Exiting.")
        return None, None

    new_data = pd.read_csv(file_path)
    new_data['finalpassage'] = new_data['finalpassage'].fillna("")
    passages = new_data['finalpassage'].tolist()
    print(f"Embedding {len(passages)} new documents...")
    embeddings = embedding_model.encode(passages, batch_size=BATCH_SIZE, show_progress_bar=True)
    return embeddings, new_data

# Main routine for updating FAISS index
def update_faiss_index():
    index = load_faiss_index()

    embeddings, new_data = process_new_data(NEW_DATA_PATH)
    if embeddings is None or new_data is None:
        return

    print("Adding new embeddings to the FAISS index...")
    index.add(embeddings)

    print(f"Saving updated FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print("Cleaning up processed data...")
    os.rename(NEW_DATA_PATH, NEW_DATA_PATH.replace("raw_data", "processed_data"))

    print("FAISS index updated successfully!")

if __name__ == "__main__":
    update_faiss_index()
