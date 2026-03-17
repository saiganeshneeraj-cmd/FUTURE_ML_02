import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_ticket_text(text):
    text = str(text).lower()
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)