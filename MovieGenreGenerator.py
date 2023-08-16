import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score

dataset = pd.read_csv('imdb.csv', delimiter=';')
dataset['genre'] = dataset['genre'].apply(lambda x: x.split(', '))
plot_summaries = dataset['plot_summary']
genres = dataset['genre']
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(genres)
unique_genres = mlb.classes_


plot_summaries = dataset['plot_summary']
vectorizer = TfidfVectorizer(max_features=1000, norm='l2')
X_tfidf = vectorizer.fit_transform(plot_summaries)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size = 0.25, random_state = 0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted_genres = mlb.inverse_transform(y_pred)
actual_genres = mlb.inverse_transform(y_test)

for i in range(len(predicted_genres)):
    print(f"Plot Summary {i+1}:")
    print("Predicted Genres:", predicted_genres[i])
    print("Actual Genres:", actual_genres[i])
    print("="*30)
hamming_loss_value = hamming_loss(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Hamming Loss: {hamming_loss_value:.4f}")
print(f"F1-Score (Micro): {f1_micro:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")