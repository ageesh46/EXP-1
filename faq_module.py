import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer, util

class FAQModule:
    def __init__(self, csv_path='data/faqs.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faq_embeddings = self.model.encode(self.df['Question'].tolist(), convert_to_tensor=True)

    def handle_faq(self, query, threshold=0.6):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.faq_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        print(f"Loading FAQ Module...")
        print(f"Ask Question : {query}")
        print(f"Best Matching FAQ : {self.df.iloc[best_idx]['Question']}")
        print(f"Similarity Score : {best_score:.2f}")

        if best_score >= threshold:
            answer = self.df.iloc[best_idx]['Answer']
            prompt = f"Answer the user query based on this official FAQ answer:\nQuery: {query}\nFAQ Answer: {answer}"
            response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        return "Sorry, I couldn't find a matching FAQ."