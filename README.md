# AI-Powered Modular E-Commerce Chatbot

Modular e-commerce assistant built using local LLMs (Ollama) and retrieval-augmented pipeline techniques.

## Features
- **Intent Recognition:** Routes user queries dynamically.
- **FAQ Module:** Uses sentence embeddings to match store policy questions.
- **Product QA Module:** Answers product queries strictly using context (RAG).
- **Recommendation Module:** Computes TF-IDF similarity to recommend products.
- **Review Intelligence Module:** Analyzes sentiment and synthesizes pros, cons, and recommendations.

## Tech Stack
- Python
- Ollama (`llama3.2:1b`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Scikit-Learn (TF-IDF Vectorizer)
- Transformers (DistilBERT Sentiment Analysis)
- Pandas

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install pandas sentence-transformers scikit-learn transformers ollama