import string
import nltk
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
# Initialize sentence-transformers model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Download necessary resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocess text: lowercase, remove punctuation, and stopwords
def preprocess_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    else:
        text = ''  # Handle non-string values (e.g., NaN)
    return text

# Create embeddings for queries or documents
def create_embeddings(documents):
    return model.encode(documents)

# Load MS-MARCO data
def load_data(file_path):
    return pd.read_csv(file_path)

import json

# After preprocessing the data and generating embeddings
if __name__ == "__main__":
    train_data = load_data('D:/RAG-GenAI/data/raw_data/train.csv')
    valid_data = load_data('D:/RAG-GenAI/data/raw_data/valid.csv')

    # Preprocess queries and passages
    train_data['query'] = train_data['query'].apply(preprocess_text)
    train_data['finalpassage'] = train_data['finalpassage'].apply(preprocess_text)
    
    valid_data['query'] = valid_data['query'].apply(preprocess_text)
    valid_data['finalpassage'] = valid_data['finalpassage'].apply(preprocess_text)

    # Create embeddings for documents
    train_embeddings = create_embeddings(train_data['finalpassage'].tolist())
    valid_embeddings = create_embeddings(valid_data['finalpassage'].tolist())

    # Save embeddings to disk
    np.save('D:/RAG-GenAI/data/processed_data/train_embeddings.npy', train_embeddings)
    np.save('D:/RAG-GenAI/data/processed_data/valid_embeddings.npy', valid_embeddings)

    # Save metadata
    metadata = {
        str(i): {"query": train_data['query'][i], "finalpassage": train_data['finalpassage'][i]}
        for i in range(len(train_data))
    }
    with open('D:/RAG-GenAI/data/processed_data/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print("Metadata saved to D:/RAG-GenAI/data/processed_data/metadata.json.")

