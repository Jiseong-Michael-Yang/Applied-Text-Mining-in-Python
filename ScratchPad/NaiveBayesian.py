# Import necessary modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Create the d
d1 = ["me free lottery", 1]
d2 = ["free get free you", 1]
d3 = ["you free scholarship", 0]
d4 = ["free to contact me", 0]
d5 = ["you won award", 0]
d6 = ["you ticket lottery", 1]

# Save the data as the dataframe
data = pd.DataFrame([d1, d2, d3, d4, d5, d6], columns=["text", "target"])

# Split the train and test data
X_train, X_test, y_train,  y_test = train_test_split(data["text"], data["target"], train_size=0.6)

# Load the vectorizers
vect_tfidf = TfidfVectorizer()
vect_count = CountVectorizer()

# Vectorize X_train
X_train_vectorized_tfidf = vect_tfidf.fit_transform(X_train)
X_train_vectorized_count = vect_count.fit_transform(X_train)

# DTM and TFIDF
features = vect_tfidf.get_feature_names()
X_train_tfidf = pd.DataFrame(X_train_vectorized_tfidf.toarray(), columns=features, index=y_train.index)
X_train_count = pd.DataFrame(X_train_vectorized_count.toarray(), columns=features, index=y_train.index)
XY_train_tfidf = pd.concat([X_train_tfidf, y_train], axis = 1)
XY_train_count = pd.concat([X_train_count, y_train], axis = 1)
XY_train_tfidf
XY_train_count
data.iloc[y_train.index,:]