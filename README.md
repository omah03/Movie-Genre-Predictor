# Movie-Genre-Predictor
A simple mini project that leverages the power of Machine Learning and the Scikit Library to roughly guess the genre of a movie based on it's plot summary. 
Currently only works with the test set, doesn't accept user input (maybe in future)

Code with explination:

Importing needed libraries.
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score`
```
Using pandas, the contents of imbd.csv are loaded into Pandas DataFrame. The delimiter = ';' argument makes note of the fact that the delimiter used in the CSV file to seperate values is ';'

We apply a lambda function that takes input x (each element in 'genre') and splits the genres listed in a single string into a list of seperate genre values.
(Basically splits multiple genres into individual genres)
```
dataset = pd.read_csv('imdb.csv', delimiter=';')
dataset['genre'] = dataset['genre'].apply(lambda x: x.split(', '))
```
We use MultiLabelBinarizer class from scikit-learn, which converts multi-label catergorical data (genres) into binary label format.
Then we use the fit_transform method of the MultiLabelBinarizer instance to convert the list into binary label representation.This results in a binary label matrix stored in y_encoded. 
Last, we obtain all the unique genres.
```
plot_summaries = dataset['plot_summary']
genres = dataset['genre']
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(genres)
unique_genres = mlb.classes_
```
Now we use the TfidVectorizer class from scikit-learn to convert text document into TF-IDF matrix. In our case, each row represnts the plot summary and each column represents a specific word from the entire dataset. We then use fit_transform to apply this transformation on the matrix of features. 
```
vectorizer = TfidfVectorizer(max_features=1000, norm='l2')
X_tfidf = vectorizer.fit_transform(plot_summaries)
```

We split the dataset into a training and a test set, matrix of features and the dependent variable.

```
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size = 0.25, random_state = 0)
```

Using Random Forest Classification model, we train it on the training set and use it to predict the test set genres. We use the inverse_transform to convert the binary genre labels to original genre names.

```
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted_genres = mlb.inverse_transform(y_pred)
actual_genres = mlb.inverse_transform(y_test)
```
We display the data nicely
```
for i in range(len(predicted_genres)):
    print(f"Plot Summary {i+1}:")
    print("Predicted Genres:", predicted_genres[i])
    print("Actual Genres:", actual_genres[i])
    print("="*30)
```

We use Hamming loss and F1 micro/macro as metrics for perforamnce. 
```
hamming_loss_value = hamming_loss(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')
```

RESULTS 
w.i.p
