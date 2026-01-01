import pandas as pd

# 3-class- positive,negative and neutral
data = {
    "text": [
        # Positive
        "I loved the movie",
        "The film was fantastic",
        "Amazing experience",
        "I really enjoyed it",
        "Absolutely brilliant",

        # Neutral
        "The movie was okay",
    "It was an average film",
    "Nothing special",
        "I feel neutral about it",
        "It was fine, nothing more",

        # Negative
        "I hated the movie",
        "The film was terrible",
        "Worst experience ever",
        "Very boring and bad",
        "I did not like it"
    ],
    "sentiment": [
        "positive","positive","positive","positive","positive",
        "neutral","neutral","neutral","neutral","neutral",
        "negative","negative","negative","negative","negative"
    ]
}

df = pd.DataFrame(data)
print(df)

# Simple preprocessing: lowercase
def clean_text(text):
    return text.lower()

df["text"] = df["text"].apply(clean_text)
print("\nAfter preprocessing:\n", df)

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform text
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

print("\nShape of TF-IDF matrix:", X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.naive_bayes import MultinomialNB

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict
y_pred_nb = nb_model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\nNaive Bayes Accuracy:", accuracy_nb)


from sklearn.linear_model import LogisticRegression

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)


# Custom inputs
custom_texts = [
    "The trip was amazing",
    "The movie was average",
    "I hated the food"
]

# Convert using TF-IDF
custom_vectors = vectorizer.transform(custom_texts)

# Predictions
pred_nb = nb_model.predict(custom_vectors)
pred_lr = lr_model.predict(custom_vectors)

print("\nPredictions using Naive Bayes:", pred_nb)
print("Predictions using Logistic Regression:", pred_lr)


