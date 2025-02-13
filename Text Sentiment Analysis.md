import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_excel('C:/Users/ALI/Desktop/IMDB Dataset.xlsx') 
df = df[['review', 'sentiment']]  

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)
df['processed_review'] = df['review'].apply(preprocess_text)

# Feature Engineering (TF-IDF Vectorization)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_review'])
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary (positive = 1, negative = 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Training - Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# Model Training - Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Predict Sentiment for User Input
def predict_sentiment(text, model, tfidf_vectorizer):
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    return 'positive' if prediction == 1 else 'negative'
user_review = input("Enter your movie review: ")

# Sentiment Prediction using Logistic Regression
predicted_sentiment_lr = predict_sentiment(user_review, model_lr, tfidf)
print(f"Sentiment Prediction using Logistic Regression: {predicted_sentiment_lr}")

# Sentiment Prediction using Naive Bayes
predicted_sentiment_nb = predict_sentiment(user_review, model_nb, tfidf)
print(f"Sen
