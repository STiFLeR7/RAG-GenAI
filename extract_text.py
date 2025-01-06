import os
import PyPDF2

def extract_text_from_pdf(pdf_path, output_file):
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += page.extract_text()
                text += "\n--- End of Page {page_num + 1} ---\n"

        # Save extracted text to a file
        with open(output_file, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        print(f"Text extracted and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred while extracting text: {e}")

if __name__ == "__main__":
    # Define paths
    pdf_path = "D:/RAG-GenAI-Dataset/Physics-WEB.pdf"
    output_file = "D:/RAG-GenAI/Physics-WEB.txt"

    # Extract text
    extract_text_from_pdf(pdf_path, output_file)
