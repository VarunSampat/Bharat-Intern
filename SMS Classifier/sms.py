from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Loading dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.dropna()

# Preprocessing the data
data['label'] = data['v1'].map({'ham': 0, 'spam': 1})
data['text'] = data['v2']
data = data.drop(['v1', 'v2'], axis=1)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorizing the data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Building the model
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Predicting the labels of the test set
y_pred = clf.predict(X_test_counts)

# Printing the accuracy of the model
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Classification Report: \n', classification_report(y_test, y_pred))