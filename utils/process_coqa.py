import json
import pandas as pd

def process_coqa_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract queries and answers
    queries, answers = [], []
    for conversation in data['data']:
        for question, answer in zip(conversation['questions'], conversation['answers']):
            queries.append(question['input_text'])  # Key for questions
            answers.append(answer['input_text'])   # Key for answers

    # Save as CSV
    processed_data = pd.DataFrame({'query': queries, 'answer': answers})
    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "D:/RAG-GenAI/data/raw_data/coqa-train-v1.0.json"  # Adjust path as necessary
    output_file = "D:/RAG-GenAI/data/fine_tuning_data_coqa.csv"
    process_coqa_dataset(input_file, output_file)
