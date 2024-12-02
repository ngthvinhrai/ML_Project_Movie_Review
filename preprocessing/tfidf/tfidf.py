import numpy as np
import pandas as pd
import re
from typing import List

class TFIDF:
    def __init__(self):
        # Từ điển lưu trữ IDF
        self.idf_dict = {}
        # Từ vựng
        self.vocabulary = []
    
    def remove_tags(self, text: str) -> np.ndarray:
        """
        Tiền xử lý văn bản
        
        Parameters:
        text (str): Văn bản đầu vào
        
        Returns:
        numpy.ndarray: Mảng các từ đã xử lý
        """
        result = re.sub(r'<.*?>','',text)           #xoá HTML tags, tiền tố r nghĩa là raw
        result = re.sub('https://.*','',result)     #xoá URLs
        result = re.sub(r'[^\w\s]', '',result)      #xoá non-alphanumeric characters ví dụ như các dấu câu
        result = result.lower()                     # Chuyển về chữ thường
        
        # Tách từ và chuyển thành numpy array
        return np.array(result.split())
    
    def fit(self, documents: List[str]) -> 'TFIDF':
        """
        Học từ vựng và tính IDF
        
        Parameters:
        documents (List[str]): Danh sách các văn bản
        
        Returns:
        self: Đối tượng hiện tại
        """
        # Xử lý văn bản thành numpy arrays
        processed_docs = [self.remove_tags(doc) for doc in documents]
        
        # Chuyển sang DataFrame để dễ xử lý
        df_docs = pd.DataFrame({'words': processed_docs})
        # Tổng số văn bản
        total_docs = len(documents)
        
        # Đếm số văn bản chứa từng từ
        word_doc_count = {}
        for doc_words in processed_docs:
            unique_words = set(doc_words)  # Chỉ đếm mỗi từ một lần trong một văn bản
            for word in unique_words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Tính IDF sử dụng numpy
        self.idf_dict = {
            word: np.log(total_docs / (count )) + 1 
            for word, count in word_doc_count.items()
        }
        
        
        # Lưu từ vựng
        self.vocabulary = list(word_doc_count.keys())
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Chuyển đổi văn bản thành ma trận TF-IDF
        
        Parameters:
        documents (List[str]): Danh sách các văn bản
        
        Returns:
        numpy.ndarray: Ma trận TF-IDF
        """
        # Khởi tạo ma trận kết quả
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        # Xử lý từng văn bản
        for i, doc in enumerate(documents):
            # Tiền xử lý văn bản
            words = self.remove_tags(doc)
            
            # Tính Term Frequency (TF) sử dụng pandas
            word_counts = pd.Series(words).value_counts()
            total_words = len(words)
            
            # Tạo vector cho văn bản
            for j, vocab_word in enumerate(self.vocabulary):
                # Tính TF
                tf = word_counts.get(vocab_word, 0) / total_words
                
                # Tính TF-IDF
                idf = self.idf_dict.get(vocab_word, 0)
                tfidf_matrix[i, j] = tf * idf
        
        return tfidf_matrix