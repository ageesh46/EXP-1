from faq_module import FAQModule
from recommendation_module import RecommendationModule
from product_module import ProductQAModule
from review_module import ReviewModule
from intent_module import detect_intent

def main():
    print("Loading modules...")
    faq_mod = FAQModule()
    rec_mod = RecommendationModule()
    prod_mod = ProductQAModule()
    rev_mod = ReviewModule()

    print("\n=== AI E-Commerce Chatbot Initialized ===")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        intent = detect_intent(user_query)
        print(f"↓\nDetected Intent: {intent}\n↓")

        if intent == 'FAQ':
            res = faq_mod.handle_faq(user_query)
            print(f"Assistant:\n{res}")
        elif intent == 'RECOMMENDATION':
            res = rec_mod.recommend(user_query)
            print(f"Assistant:\n{res}")
        elif intent == 'PRODUCT':
            res = prod_mod.answer_question(user_query)
            print(f"Assistant:\n{res}")
        elif intent == 'REVIEW':
            res = rev_mod.analyze_product_reviews(user_query)
            print(f"Assistant:\n{res}")

if __name__ == '__main__':
    main()