import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationModule:
    def __init__(self, csv_path='data/products.csv'):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Description'].fillna(''))

    def recommend(self, query, top_n=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        retrieved_products = self.df.iloc[top_indices]

        print("Retrieved Products:")
        for idx, row in retrieved_products.iterrows():
            print(f"- {row['ProductName']}")

        context = retrieved_products[['ProductName', 'Brand', 'Price', 'Description']].to_string()
        prompt = f"User Request: {query}\n\nTop Products:\n{context}\n\nRecommend the single best product among these and give a concise reason."
        
        response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']