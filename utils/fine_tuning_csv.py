import pandas as pd

# Load the original dataset
data = pd.read_csv("D:/RAG-GenAI/data/raw_data/train.csv")

# Generate query-answer pairs
fine_tuning_data = data[['query', 'answers']].dropna()  # Adjust column names if needed
fine_tuning_data.rename(columns={'query': 'query', 'answers': 'answer'}, inplace=True)

# Save to CSV
fine_tuning_data.to_csv("D:/RAG-GenAI/data/fine_tuning_data.csv", index=False)
print("fine_tuning_data.csv created successfully!")
