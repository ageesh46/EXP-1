import pandas as pd
import ollama
from transformers import pipeline

class ReviewModule:
    def __init__(self, csv_path='data/reviews.csv'):
        self.df = pd.read_csv(csv_path)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze_product_reviews(self, product_name):
        matched_reviews = self.df[self.df['ProductName'].str.contains(product_name, case=False, na=False)]
        if matched_reviews.empty:
            return f"No reviews found for '{product_name}'."

        reviews_list = matched_reviews['Review'].tolist()
        results = self.sentiment_analyzer(reviews_list)

        pos_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        neg_count = len(results) - pos_count

        print(f"Positive : {pos_count} | Negative : {neg_count}")
        
        context = "\n".join([f"- {rev}" for rev in reviews_list])
        prompt = f"Product: {product_name}\nCustomer Reviews:\n{context}\n\nSummarize the Overall Opinion, Strengths, Weaknesses, and give a concise Buying Suggestion."

        response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']