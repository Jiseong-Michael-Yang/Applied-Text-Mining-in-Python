from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

d1 = ["me free lottery", 1]
d2 = ["free get free you", 1]
d3 = ["you free scholarship", 0]
d4 = ["free to contact me", 0]
d5 = ["you won award", 0]
d6 = ["you ticket lottery", 1]

data = pd.DataFrame([d1, d2, d3, d4, d5, d6], columns=["text", "target"])
print(data)

X_train, X_test, y_train,  y_test = train_test_split(data["text"], data["target"], train_size=0.6, random_state = 42)

vect = TfidfVectorizer()
X_train_vectorized = vect.fit_transform(X_train)