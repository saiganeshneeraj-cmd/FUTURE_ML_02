import joblib
from preprocess import clean_ticket_text

# 1. Load the saved "brains"
print("Loading models...")
cat_model = joblib.load('category_model.pkl')
prio_model = joblib.load('priority_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def solve_ticket():
    print("\n--- AI Support Ticket Classifier ---")
    user_input = input("Enter ticket description (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        return False

    # 2. Preprocess & Vectorize
    cleaned = clean_ticket_text(user_input)
    vectorized = tfidf.transform([cleaned])
    
    # 3. Predict
    category = cat_model.predict(vectorized)[0]
    priority = prio_model.predict(vectorized)[0]
    
    print(f"\nResult:")
    print(f" > Predicted Category: {category}")
    print(f" > Predicted Priority: {priority}")
    return True

if __name__ == "__main__":
    running = True
    while running:
        running = solve_ticket()