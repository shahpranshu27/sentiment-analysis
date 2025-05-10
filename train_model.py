import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset (in a real project, use a larger dataset)
data = {
    'text': [
        "I love this product!",
        "This is terrible.",
        "Awesome experience!",
        "Worst service ever.",
        "Highly recommended!",
        "I hate it."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')