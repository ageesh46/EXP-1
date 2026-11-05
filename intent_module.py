import ollama

def detect_intent(query):
    prompt = f"""Classify the user intent into EXACTLY ONE of these categories: FAQ, RECOMMENDATION, PRODUCT, REVIEW.
Query: "{query}"
Output only the category name."""
    response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
    intent = response['message']['content'].strip().upper()
    
    for category in ['FAQ', 'RECOMMENDATION', 'PRODUCT', 'REVIEW']:
        if category in intent:
            return category
    return 'FAQ'