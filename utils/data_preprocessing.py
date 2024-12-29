import string
import nltk
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize sentence-transformers model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Download necessary resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocess text: lowercase, remove punctuation, and stopwords
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Create embeddings for queries or documents
def create_embeddings(documents):
    return model.encode(documents)

# Load MS-MARCO data
def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    train_data = load_data('data/raw_data/train.csv')
    valid_data = load_data('data/raw_data/valid.csv')

    # Preprocess queries and documents
    train_data['query'] = train_data['query'].apply(preprocess_text)
    train_data['passage'] = train_data['passage'].apply(preprocess_text)

    valid_data['query'] = valid_data['query'].apply(preprocess_text)
    valid_data['passage'] = valid_data['passage'].apply(preprocess_text)

    # Create embeddings for documents
    train_embeddings = create_embeddings(train_data['passage'].tolist())
    valid_embeddings = create_embeddings(valid_data['passage'].tolist())

    # Save embeddings if needed
    # You can save embeddings to disk for later use (e.g., using pickle or numpy)
