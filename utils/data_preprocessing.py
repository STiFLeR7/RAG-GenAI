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

if __name__ == "__main__":
    train_data = load_data('data/raw_data/train.csv')
    valid_data = load_data('data/raw_data/valid.csv')

    # Preprocess queries and passages
    train_data['query'] = train_data['query'].apply(preprocess_text)
    train_data['finalpassage'] = train_data['finalpassage'].apply(preprocess_text)  # Correct column name for document

    valid_data['query'] = valid_data['query'].apply(preprocess_text)
    valid_data['finalpassage'] = valid_data['finalpassage'].apply(preprocess_text)  # Correct column name for document

    # Create embeddings for documents
    train_embeddings = create_embeddings(train_data['finalpassage'].tolist())
    valid_embeddings = create_embeddings(valid_data['finalpassage'].tolist())

    # Optionally, save embeddings to disk for later use (e.g., with pickle or numpy)
