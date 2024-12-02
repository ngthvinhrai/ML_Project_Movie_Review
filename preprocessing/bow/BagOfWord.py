import numpy as np
import pandas as pd
import re
import requests

class BagOfWord:
    def __init__(self):
        """Khởi tạo class."""
        self.vocabulary = []  # Danh sách từ vựng duy nhất
        self.bag_vector = None  # Ma trận Bag-of-Words
        self.stopwords = self.load_stopwords()  # Danh sách từ dừng

    @staticmethod
    def load_stopwords():
        """Tải danh sách từ dừng từ một nguồn trực tuyến."""
        url = ("https://gist.githubusercontent.com/sebleier/554280/raw/""7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords")
        response = requests.get(url)
        return response.text.splitlines()

    @staticmethod
    def remove_tags(text):
        """Loại bỏ các thẻ HTML, URL, ký tự đặc biệt và chuyển thành chữ thường."""
        text = re.sub(r'<.*?>', '', text)  # Loại bỏ thẻ HTML
        text = re.sub('https?://\S+', '', text)  # Loại bỏ URL
        text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
        return text.lower()  # Chuyển thành chữ thường

    def preprocess(self, text):
        """Tiền xử lý văn bản."""
        text = self.remove_tags(text)  # Làm sạch văn bản
        text = ' '.join([word for word in text.split() if word not in self.stopwords])  # Loại bỏ từ dừng
        return text

    def fit(self, dataset):
        # Tiền xử lý dữ liệu
        dataset = [self.preprocess(text) for text in dataset]

        # Xây dựng từ vựng
        tokenized_data = [text.split() for text in dataset]
        self.vocabulary = sorted(set(word for words in tokenized_data for word in words))

        # Tạo ma trận Bag-of-Words
        self.bag_vector = np.zeros((len(dataset), len(self.vocabulary)), dtype=int)

        # Điền số lần xuất hiện của từ
        for idx, words in enumerate(tokenized_data):
            for word in words:
                if word in self.vocabulary:
                    word_idx = self.vocabulary.index(word)
                    self.bag_vector[idx][word_idx] += 1

        return self.bag_vector

    def transform(self, dataset):
        """
        Chuyển tập dữ liệu mới thành ma trận Bag-of-Words dựa trên từ vựng đã có.
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Call `fit` first.")

        # Tiền xử lý dữ liệu
        dataset = [self.preprocess(text) for text in dataset]

        # Tạo ma trận Bag-of-Words
        bag_vector = np.zeros((len(dataset), len(self.vocabulary)), dtype=int)

        for idx, text in enumerate(dataset):
            words = text.split()
            for word in words:
                if word in self.vocabulary:
                    word_idx = self.vocabulary.index(word)
                    bag_vector[idx][word_idx] += 1

        return bag_vector

# Đọc dữ liệu từ file CSV
df= pd.read_csv('/content/IMDB Dataset.csv')

# Lấy cột "review"
reviews = df['review'].tolist()  # Chuyển cột "review" thành danh sách

# Khởi tạo và xử lý
bow = BagOfWord()

# Xây dựng từ vựng và ma trận Bag-of-Words từ 100 dòng đầu tiên
bag_vector = bow.fit(reviews[:50])  # Sử dụng 100 dòng đầu tiên để demo
print("Vocabulary:", bow.vocabulary)
print("Bag-of-Words Matrix (Fit):\n", bag_vector)

# Chuyển đổi dữ liệu mới
new_reviews = ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due"]
new_bag_vector = bow.transform(new_reviews)
print("Bag-of-Words Matrix (Transform):\n", new_bag_vector)