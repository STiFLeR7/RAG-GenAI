import os
import pickle
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_file, output_file, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for the processed text chunks and save them to a file.

    Args:
        input_file (str): Path to the input text file containing processed chunks.
        output_file (str): Path to save the embeddings.
        model_name (str): Name of the SentenceTransformers model to use.
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    try:
        # Load the embedding model
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Read chunks from the input file
        with open(input_file, 'r', encoding='utf-8') as file:
            chunks = []
            current_chunk = []
            for line in file:
                if line.startswith("--- Chunk"):
                    if current_chunk:
                        chunks.append(" ".join(current_chunk).strip())
                        current_chunk = []
                else:
                    current_chunk.append(line.strip())
            if current_chunk:  # Add the last chunk
                chunks.append(" ".join(current_chunk).strip())

        print(f"Total chunks to embed: {len(chunks)}")

        # Generate embeddings
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

        # Save embeddings and chunks
        with open(output_file, 'wb') as file:
            pickle.dump({"chunks": chunks, "embeddings": embeddings}, file)

        print(f"Embeddings saved to {output_file}")

    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")

if __name__ == "__main__":
    # Define paths
    input_file = "D:/RAG-GenAI/Physics-WEB-processed.txt"
    output_file = "D:/RAG-GenAI/Physics-embeddings.pkl"

    # Generate embeddings
    generate_embeddings(input_file, output_file)
