import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = [
    "You're so dumb and ugly",
    "I hate you",
    "You're amazing!",
    "Let's be friends",
    "Get lost loser",
    "You're cool"
]
labels = [1, 1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)
joblib.dump((model, vectorizer), "model.pkl")
print("✅ Model and vectorizer saved as model.pkl.")

