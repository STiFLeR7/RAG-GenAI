from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model_and_tokenizer(model_dir):
    """
    Load the fine-tuned T5 model and tokenizer from the specified directory.
    """
    print("Loading fine-tuned model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_response(model, tokenizer, query):
    """
    Generate a response for the given query using the fine-tuned model.
    """
    input_text = f"Answer this question: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Specify the directory where the fine-tuned model is saved
    model_dir = "../models/t5_fine_tuned"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # Example queries for testing
    queries = [
        "What is artificial intelligence?",
        "Explain the importance of renewable energy.",
        "How does photosynthesis work?",
    ]

    # Generate and display responses
    for query in queries:
        print(f"\nQuery: {query}")
        response = generate_response(model, tokenizer, query)
        print(f"Response: {response}")
