import requests
import re

def load_stopwords():
    url = ("https://gist.githubusercontent.com/sebleier/554280/raw/""7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords")
    response = requests.get(url)
    return response.text.splitlines()

def remove_tags(text):
    text = re.sub(r'<.*?>', '', text)  # Loại bỏ thẻ HTML
    text = re.sub('https?://\S+', '', text)  # Loại bỏ URL
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    return text.lower()  # Chuyển thành chữ thường

def preprocess(text, stopwords):
    text = remove_tags(text)  # Làm sạch văn bản
    text = ' '.join([word for word in text.split() if word not in stopwords])  # Loại bỏ từ dừng
    return text