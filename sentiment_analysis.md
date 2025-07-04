Here is a **detailed explanation of every step** involved in performing **Sentiment Analysis** using **Naive Bayes** with **NLTK**, **Corpus**, **Bag of Words (BoW)**, and **TF-IDF**.

---

## 🧠 Project: Sentiment Analysis using Naive Bayes (with NLTK + BoW + TF-IDF)

---

### 🔧 Step 1: Install Required Libraries

To begin, install the necessary Python libraries:

```bash
pip install pandas scikit-learn nltk
```

* `pandas`: For loading and manipulating the dataset.
* `nltk`: Natural Language Toolkit, used for text preprocessing.
* `scikit-learn`: For machine learning (Naive Bayes, vectorizers, evaluation).

---

### 📥 Step 2: Load Your Dataset

Assuming your CSV file has two columns: `comments` and `review`.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv("sentiment_analysis_dataset.csv")

# Display first few rows
print(df.head())
```

---

### 🧹 Step 3: Text Preprocessing using NLTK

Text data needs to be cleaned before using in models.

#### Tasks:

1. Convert to lowercase
2. Remove punctuation
3. Tokenize text
4. Remove stopwords
5. Apply stemming

```python
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize stemmer and stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem
    return ' '.join(words)

# Apply preprocessing to comments
df['clean_comments'] = df['comments'].apply(preprocess)
```

---

### 🧠 Step 4: Split Dataset into Training and Testing

We need to train the model on one portion of data and test it on another.

```python
from sklearn.model_selection import train_test_split

X = df['clean_comments']
y = df['review']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 📦 Step 5A: Vectorization using Bag of Words (BoW)

Convert text to numerical format using **CountVectorizer** (BoW method).

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
```

---

### 📦 Step 5B: Vectorization using TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) weighs words more meaningfully than BoW.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

---

### 🤖 Step 6: Train Naive Bayes Model

Use `MultinomialNB`, the best Naive Bayes classifier for text data.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)  # Use X_train_bow if using BoW instead
```

---

### 🧪 Step 7: Predict and Evaluate the Model

Check how well the model is performing on unseen test data.

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)  # Or X_test_bow for BoW

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed classification performance
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

### 📊 Output Example:

```
Accuracy: 0.91

Classification Report:
              precision    recall  f1-score   support

        bad       0.89      0.88      0.89       65
      good       0.93      0.92      0.92       78
     worst       0.91      0.92      0.91       57

avg / total       0.91      0.91      0.91      200
```

---

### 📝 Final Notes:

| Concept       | Details                                     |
| ------------- | ------------------------------------------- |
| Text Cleaning | Remove noise, stopwords, and apply stemming |
| BoW           | Simple word frequency counts                |
| TF-IDF        | Adjusts word weights by importance          |
| Naive Bayes   | Probabilistic classifier good for text      |
| Evaluation    | Accuracy + Precision, Recall, F1-score      |

---

Would you like this as a `.py` file or Jupyter Notebook to download?
