import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load preprocessed data
X_tfidf = joblib.load('X_tfidf.pkl')
y = joblib.load('y.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Save splits for evaluation
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
joblib.dump(lr, 'lr_model.pkl')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'rf_model.pkl')

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
joblib.dump(nb, 'nb_model.pkl')

print("Basic models trained and saved: lr_model.pkl, rf_model.pkl, nb_model.pkl")
