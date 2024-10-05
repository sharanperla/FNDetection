import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')

# Load the datasets from the data folder
true_news = pd.read_csv('../data/True.csv')
fake_news = pd.read_csv('../data/Fake.csv')

# Add a label to each dataset (1 for real news, 0 for fake news)
true_news['label'] = 1
fake_news['label'] = 0

# Combine datasets
data = pd.concat([true_news, fake_news])

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Preprocess text (lowercase, remove punctuation, stopwords)
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])

data['text'] = data['text'].apply(preprocess_text)

# Split dataset into features and labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, y_train)

# Train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)

# Evaluate models
log_pred = log_model.predict(X_test_tfidf)
dt_pred = dt_model.predict(X_test_tfidf)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred)}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred)}")

# Save the models and vectorizer to the src folder
with open('../src/log_model.pkl', 'wb') as log_file:
    pickle.dump(log_model, log_file)

with open('../src/dt_model.pkl', 'wb') as dt_file:
    pickle.dump(dt_model, dt_file)

with open('../src/tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf, vec_file)
