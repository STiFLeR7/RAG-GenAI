import os
import re

def preprocess_text(input_file, output_file, chunk_size=1000):
    
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters

        # Split text into chunks
        chunks = []
        while len(text) > chunk_size:
            # Find the nearest sentence boundary
            split_index = text[:chunk_size].rfind('. ')
            if split_index == -1:  # If no sentence boundary is found
                split_index = chunk_size
            chunks.append(text[:split_index + 1].strip())
            text = text[split_index + 1:].strip()
        if text:
            chunks.append(text.strip())

        # Save chunks to the output file
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, chunk in enumerate(chunks):
                file.write(f"--- Chunk {i + 1} ---\n")
                file.write(chunk + "\n\n")

        print(f"Preprocessed text saved to {output_file}. Total chunks: {len(chunks)}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    # Define paths
    input_file = "D:/RAG-GenAI/Physics-WEB.txt"
    output_file = "D:/RAG-GenAI/Physics-WEB-processed.txt"

    # Preprocess the text
    preprocess_text(input_file, output_file)
