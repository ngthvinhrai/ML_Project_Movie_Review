import numpy as np
import pandas as pd
from preprocessing.preprocess_data import load_stopwords, remove_tags, preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    df = pd.read_csv('E:/Project/ML_Project_Movie_Review/IMDB Dataset.csv')

    stopwords = load_stopwords()
    df['review'] = df['review'].apply(lambda x: remove_tags(x))
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    encoder = LabelEncoder()
    encoded_label = encoder.fit_transform(df['sentiment'].values)
    encoded_label[encoded_label==0] = -1
    
    train_dataset, test_dataset, train_label, test_label = train_test_split(df, encoded_label, test_size=0.4, random_state=42)

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(df['review'])

    X_train_count = count_vectorizer.transform(train_dataset)
    print(X_train_count[0])