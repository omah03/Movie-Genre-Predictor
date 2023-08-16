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
