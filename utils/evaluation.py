import torch
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

# Custom Dataset Class
class T5Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.targets[idx],
        }

def calculate_precision_recall(retrieved_indices, relevant_indices, k=5):
    """
    Calculate Precision@k and Recall@k for retrieval.

    Args:
        retrieved_indices (list): Indices of retrieved documents.
        relevant_indices (list): Indices of relevant documents.
        k (int): Number of top documents to consider.

    Returns:
        precision (float): Precision@k.
        recall (float): Recall@k.
    """
    relevant_retrieved = [idx for idx in retrieved_indices[:k] if idx in relevant_indices]
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 0
    return precision, recall

def calculate_mrr(retrieved_indices, relevant_indices):
    """
    Calculate Mean Reciprocal Rank (MRR) for retrieval.

    Args:
        retrieved_indices (list): Indices of retrieved documents.
        relevant_indices (list): Indices of relevant documents.

    Returns:
        mrr (float): Mean Reciprocal Rank.
    """
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1 / rank
    return 0

def calculate_bleu(reference, generated):
    """
    Calculate BLEU score for generated text.

    Args:
        reference (str): Reference text.
        generated (str): Generated text.

    Returns:
        bleu_score (float): BLEU score.
    """
    return sentence_bleu([reference.split()], generated.split())

def calculate_rouge(reference, generated):
    """
    Calculate ROUGE-1 and ROUGE-L scores for generated text.

    Args:
        reference (str): Reference text.
        generated (str): Generated text.

    Returns:
        dict: Dictionary containing ROUGE-1 and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

def fine_tune_t5(data_path, output_dir):
    """
    Fine-tune T5 model for text generation.

    Args:
        data_path (str): Path to the CSV file containing 'query' and 'answer' columns.
        output_dir (str): Directory to save the fine-tuned model.
    """
    # Load dataset
    data = pd.read_csv(data_path)

    # Split into training and validation sets
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # Tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Prepare training data
    train_inputs = tokenizer(train_data['query'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt").input_ids
    train_targets = tokenizer(train_data['answer'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt").input_ids
    train_dataset = T5Dataset(train_inputs, train_targets)

    # Prepare evaluation data
    eval_inputs = tokenizer(eval_data['query'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt").input_ids
    eval_targets = tokenizer(eval_data['answer'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt").input_ids
    eval_dataset = T5Dataset(eval_inputs, eval_targets)

    # Define training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,  # Log every 50 steps
        evaluation_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving the fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuned T5 model saved!")

if __name__ == "__main__":
    # Example Usage for Metrics
    retrieved_indices = [0, 2, 3, 4, 5]
    relevant_indices = [2, 3]

    precision, recall = calculate_precision_recall(retrieved_indices, relevant_indices, k=3)
    mrr = calculate_mrr(retrieved_indices, relevant_indices)

    print(f"Precision@k: {precision}, Recall@k: {recall}, MRR: {mrr}")

    reference_text = "The cat sat on the mat."
    generated_text = "A cat was sitting on a mat."

    bleu = calculate_bleu(reference_text, generated_text)
    rouge = calculate_rouge(reference_text, generated_text)

    print(f"BLEU: {bleu}")
    print(f"ROUGE: {rouge}")

    # Example Usage for Fine-Tuning
    data_path = "D:/RAG-GenAI/data/fine_tuning_data.csv"
    output_dir = "D:/RAG-GenAI/models/t5_fine_tuned"

    fine_tune_t5(data_path, output_dir)
