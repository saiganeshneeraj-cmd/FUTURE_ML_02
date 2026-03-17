import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import clean_ticket_text

print("Step 1: Loading dataset...")
df = pd.read_csv('data/customer_support_tickets.csv')

print("Step 2: Cleaning text...")
df['cleaned_text'] = df['Ticket Description'].apply(clean_ticket_text)

print("Step 3: Vectorizing text...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_text'])

print("Step 4: Training Category Classifier...")
y_cat = df['Ticket Type']
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
cat_model.fit(X_train, y_train)

print("Step 5: Training Priority Classifier...")
y_prio = df['Ticket Priority']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_prio, test_size=0.2, random_state=42)
prio_model = RandomForestClassifier(n_estimators=100, random_state=42)
prio_model.fit(X_train_p, y_train_p)

print("\n--- RESULTS ---")
print("Category Report:\n", classification_report(y_test, cat_model.predict(X_test)))

joblib.dump(cat_model, 'category_model.pkl')
joblib.dump(prio_model, 'priority_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSuccess! Models saved.")