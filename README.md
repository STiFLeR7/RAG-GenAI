
# **RAG-GenAI**

**RAG-GenAI** is an advanced Retrieval-Augmented Generation (RAG) system designed to redefine the boundaries of generative AI by combining state-of-the-art document retrieval techniques with **natural language processing** (NLP). It leverages cutting-edge tools like **NLTK**, **rouge-score**, and **sentence-transformers** to ensure precise, contextually enriched, and factually grounded responses for diverse applications.


## 🚀 Overview
Generative AI models often face challenges such as:

• Generating hallucinated information. 

• Providing outdated or irrelevant data.

• Lacking integration with real-time or custom knowledge bases.

RAG-GenAI overcomes these barriers by fusing retrieval pipelines with the generation capabilities of large language models (LLMs), creating outputs that are factually accurate and context-aware.

# Key Technologies

• **NLTK (Natural Language Toolkit)**: Used for tokenization,      stemming, and linguistic preprocessing, enhancing text parsing and content understanding.

• **Rouge-Score**: Implements robust evaluation metrics to assess the quality of generated responses against reference texts.

• **Sentence-Transformers**: Enables semantic similarity and vector-based document retrieval, ensuring the most contextually relevant information is retrieved.

• **TF-IDF & BM25**: Incorporates traditional retrieval methods to augment dense embeddings with keyword-based relevance.

• **Pretrained Models**: Utilizes cutting-edge models like BERT, DistilBERT, and GPT variants for retrieval and generation tasks.

• **PyTorch and Hugging Face Transformers**: Frameworks supporting the modular design and fine-tuning of both retrieval and generative models.
## ✨ Features
• **Hybrid Document Retrieval**: Combines dense vector embeddings (via sentence-transformers) with traditional TF-IDF and BM25 for optimal results.

• **Customizable Pipelines**: Fine-tune retrieval and generation processes for domain-specific use cases.

• **Quality Assurance**: Includes metrics like BLEU, ROUGE, and METEOR for evaluating the precision and coherence of generated outputs.

• **Plug-and-Play Design**: Easily integrate your custom datasets and models with minimal effort.
## 🔧 Installation
1. Clone the repository:
```git clone https://github.com/STiFLeR7/RAG-GenAI.git ```

```cd RAG-GenAI```

2. Install Dependencies
```pip install -r requirements.txt  ```
## 📚 Usage
**Step 1: Configure Retrieval and Generation**

Modify the ```config.py``` file to specify:

• Data source paths.

• Retrieval models (e.g., sentence-transformers or BM25).

• Generative models (e.g., GPT-3, LLaMA, or T5).

**Step 2: Prepare the Document Corpus**

Ensure your document repository is structured and preprocessed using NLTK for tokenization, stopword removal, and lemmatization.

**Step 3: Run the Main Script**

Launch the system to process queries and generate responses:

```python main.py```
## 🧠 Model Workflow

**1.** **Query Processing:**

• Tokenized and processed using NLTK to extract key phrases and improve retrieval accuracy.

**2. Document Retrieval:**

• Uses sentence-transformers to perform dense vector searches.

• Supplements with BM25 or TF-IDF for hybrid retrieval.

**3. Response Generation:**

• Inputs retrieved documents into the generative model.

• Produces responses with contextual relevance and factual grounding.

**4. Evaluation:**

• Evaluates the quality of responses using metrics like ROUGE, BLEU, and METEOR.
## Training Summary

• **Training Loss:** ~0.272

• **Evaluation Loss:** ~0.383

• **Runtime:**

 • Training: 3 hours, 25 minutes

 • Evaluation: 212 seconds (~3.5 minutes)
## 🌟 Evaluation
### Metrics Used:

• **ROUGE-Score**: Measures overlap between generated and reference texts.

• **BLEU**: Evaluates the n-gram precision of generated content.

• **METEOR**: Assesses linguistic diversity and synonym-based matching.

These metrics ensure high-quality responses, improving both fluency and informativeness.
## 🤝 Contributing

We welcome your contributions to improve RAG-GenAI:

1. Fork the repository and create a feature branch:

    ```git checkout -b feature-name  ```

2. Commit your changes and push:

    ```git commit -m "Add a feature or fix" ```  
    ```git push origin feature-name ```

3. Submit a pull request for review.  

## 📜 License

This project is licensed under the ***MIT License***—open for use and modification.
## 🔗 Acknowledgments

Special thanks to the developers of **NLTK**, **Hugging Face Transformers**, and **sentence-transformers**, whose tools form the backbone of this system.