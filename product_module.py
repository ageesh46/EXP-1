import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer, util

class ProductQAModule:
    def __init__(self, csv_path='data/products.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_texts = (self.df['ProductName'] + " " + self.df['Brand'] + " " + self.df['Category']).tolist()
        self.embeddings = self.model.encode(self.product_texts, convert_to_tensor=True)

    def answer_question(self, query):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        best_idx = scores.argmax().item()
        product = self.df.iloc[best_idx]

        print(f"Retrieved Product: {product['ProductName']}")
        print(f"Similarity : {scores[best_idx].item():.2f}")

        context = f"ProductName: {product['ProductName']}, Brand: {product['Brand']}, Price: ₹{product['Price']}, Category: {product['Category']}, Description: {product['Description']}"
        prompt = f"Context: {context}\nQuestion: {query}\n\nAnswer ONLY using the context provided above. Be direct and concise."

        # UPDATED TO 'llama3'
        response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']