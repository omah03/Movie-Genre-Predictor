# Movie-Genre-Predictor
A simple mini project that leverages the power of Machine Learning and the Scikit Library to roughly guess the genre of a movie based on it's plot summary. 
Currently only works with the test set, doesn't accept user input

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


