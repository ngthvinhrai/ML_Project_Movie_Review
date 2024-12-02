import numpy as np
import pandas as pd
import re
import requests
from collections import Counter
class BagOfWords:
    def __init__(self, rows=1):
        self.vocabulary = []  # Lưu trữ từ vựng
        self.rows = rows  # Số dòng cần xử lý
        self.tokenized_data = []  # Lưu trữ danh sách từ đã tách của các dòng

    def fit(self, df, column_name):
        def tokenize(paragraph):
            return paragraph.split()  # Tách từ

        # Lấy `self.rows` dòng đầu tiên và lưu danh sách từ
        partial_df = df.head(self.rows)
        self.tokenized_data = partial_df[column_name].apply(tokenize).tolist()

        # Xây dựng từ vựng
        self.vocabulary = list(set(word for words_list in self.tokenized_data for word in words_list))
        self.vocabulary.sort()  # Đảm bảo từ vựng được sắp xếp theo bảng chữ cái

    def transform(self):
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Call FIT")

        # Khởi tạo ma trận BoW
        bag_matrix = np.zeros((len(self.tokenized_data), len(self.vocabulary)), dtype=int)

        for idx, words in enumerate(self.tokenized_data):
            for word in words:
                if word in self.vocabulary:  # Kiểm tra từ có trong từ vựng
                    index = self.vocabulary.index(word)  # Vị trí từ trong từ vựng
                    bag_matrix[idx][index] += 1  # Tăng số lần xuất hiện của từ trong BoW vector

        return bag_matrix